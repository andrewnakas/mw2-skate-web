//! Dirty-room export of one MW2 map for the clean-room pipeline
//! (skate3-cleanroom `games/mw2`). The output holds retail data: keep it
//! private, never commit or publish it.
//!
//!   IW4L_GAMES=<games root> clean_export <zone> <out dir>
//!
//! Writes, all in the skate engine's space (Y up, metres):
//!   export.json    zone, surface_names, spawns [[x, y, z, heading]], materials
//!   render.npz     positions normals uv material (1-based) indices
//!   collision.npz  triangles (K,3,3) surface (IW4 surface type index)
//!   images/*.png   colour maps, decoded to RGBA8
//!
//! Render geometry is the world batches plus every static model's LOD0 baked
//! into world space. Collision is every player-solid brush (as polygons cut
//! from its planes) plus the clip mesh triangles.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use bevy::prelude::{Mat3, Vec3};
use bevy::render::mesh::{Indices, VertexAttributeValues};

const INCHES: f32 = 0.0254;
/// `MASK_PLAYER_SOLID` (sim/src/bullet_collision.rs).
const MASK_PLAYER_SOLID: u32 = 0x0281_0011;
/// IW4 surface type names, by the index in surface flag bits 20..25.
const SURFACE_NAMES: [&str; 31] = [
    "default", "bark", "brick", "carpet", "cloth", "concrete", "dirt", "flesh", "foliage", "glass",
    "grass", "gravel", "ice", "metal", "mud", "paper", "plaster", "rock", "sand", "snow", "water",
    "wood", "asphalt", "ceramic", "plastic", "rubber", "cushion", "fruit", "paintedmetal",
    "riotshield", "slush",
];

fn surface_type(flags: u32) -> u32 {
    ((flags >> 20) & 0x1f).min(SURFACE_NAMES.len() as u32 - 1)
}

/// IW4 (Z up, inches) to the skate engine (Y up, metres).
fn position(p: [f32; 3]) -> [f32; 3] {
    [p[0] * INCHES, p[2] * INCHES, -p[1] * INCHES]
}

fn direction(d: [f32; 3]) -> [f32; 3] {
    let v = Vec3::new(d[0], d[2], -d[1]).normalize_or(Vec3::Y);
    v.to_array()
}

fn fatal(message: &str) -> ! {
    eprintln!("clean_export: {message}");
    std::process::exit(1)
}

#[derive(Default)]
struct Render {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uv: Vec<[f32; 2]>,
    material: Vec<u32>,
    indices: Vec<u32>,
}

impl Render {
    /// Append a triangle-list mesh, transformed into IW4 world space by
    /// `origin + scale * basis * p`, then converted to engine space.
    fn push(&mut self, mesh: &bevy::prelude::Mesh, material: u32, basis: Mat3, origin: Vec3, scale: f32) -> bool {
        let Some(VertexAttributeValues::Float32x3(pos)) = mesh.attribute(bevy::prelude::Mesh::ATTRIBUTE_POSITION) else {
            return false;
        };
        let normals = match mesh.attribute(bevy::prelude::Mesh::ATTRIBUTE_NORMAL) {
            Some(VertexAttributeValues::Float32x3(n)) if n.len() == pos.len() => Some(n),
            _ => None,
        };
        let uvs = match mesh.attribute(bevy::prelude::Mesh::ATTRIBUTE_UV_0) {
            Some(VertexAttributeValues::Float32x2(u)) if u.len() == pos.len() => Some(u),
            _ => None,
        };
        let Some(indices) = mesh.indices() else { return false };
        let indices: Vec<u32> = match indices {
            Indices::U16(v) => v.iter().map(|&i| u32::from(i)).collect(),
            Indices::U32(v) => v.clone(),
        };
        if indices.len() % 3 != 0 || indices.iter().any(|&i| i as usize >= pos.len()) {
            return false;
        }
        let base = self.positions.len() as u32;
        for (i, p) in pos.iter().enumerate() {
            let world = origin + basis * (Vec3::from_array(*p) * scale);
            if !world.is_finite() {
                return false;
            }
            self.positions.push(position(world.to_array()));
            let n = normals.map_or(Vec3::Z, |n| basis * Vec3::from_array(n[i]));
            self.normals.push(direction(n.to_array()));
            self.uv.push(uvs.map_or([0.0, 0.0], |u| u[i]));
            self.material.push(material);
        }
        self.indices.extend(indices.iter().map(|i| base + i));
        true
    }
}

/// Faces of a convex brush (`n.p <= d` is solid) as outward-wound triangles in IW4 space.
fn brush_triangles(planes: &[[f32; 4]], out: &mut Vec<([[f32; 3]; 3], usize)>) {
    const BIG: f32 = 65536.0;
    const EPS: f32 = 0.01;
    for (i, plane) in planes.iter().enumerate() {
        let n = Vec3::new(plane[0], plane[1], plane[2]);
        if n.length_squared() < 0.5 {
            continue;
        }
        // Skip a plane that repeats an earlier one.
        if planes[..i].iter().any(|q| (Vec3::new(q[0], q[1], q[2]) - n).length() < 1e-4 && (q[3] - plane[3]).abs() < 1e-3) {
            continue;
        }
        let up = if n.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
        let u = n.cross(up).normalize();
        let v = n.cross(u);
        let c = n * plane[3];
        // Counter-clockwise seen from outside (+n).
        let mut poly = vec![c - u * BIG - v * BIG, c + u * BIG - v * BIG, c + u * BIG + v * BIG, c - u * BIG + v * BIG];
        if u.cross(v).dot(n) < 0.0 {
            poly.reverse();
        }
        for (j, other) in planes.iter().enumerate() {
            if j == i || poly.len() < 3 {
                continue;
            }
            let m = Vec3::new(other[0], other[1], other[2]);
            let d = other[3];
            let mut next = Vec::with_capacity(poly.len() + 1);
            for k in 0..poly.len() {
                let a = poly[k];
                let b = poly[(k + 1) % poly.len()];
                let da = m.dot(a) - d;
                let db = m.dot(b) - d;
                if da <= EPS {
                    next.push(a);
                }
                if (da < -EPS && db > EPS) || (da > EPS && db < -EPS) {
                    next.push(a + (b - a) * (da / (da - db)));
                }
            }
            poly = next;
        }
        if poly.len() < 3 {
            continue;
        }
        for k in 1..poly.len() - 1 {
            let t = [poly[0], poly[k], poly[k + 1]];
            if (t[1] - t[0]).cross(t[2] - t[0]).length() > 1e-3 {
                out.push((t.map(|p| p.to_array()), i));
            }
        }
    }
}

/// A minimal .npy (v1.0, little endian) for `zip` entries.
fn npy(descr: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let shape = match shape {
        [n] => format!("({n},)"),
        _ => format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend((header.len() as u16).to_le_bytes());
    out.extend(header.as_bytes());
    out.extend(data);
    out
}

fn f32_bytes<const N: usize>(v: &[[f32; N]]) -> Vec<u8> {
    v.iter().flatten().flat_map(|x| x.to_le_bytes()).collect()
}

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn write_npz(path: &Path, arrays: &[(&str, Vec<u8>)]) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for (name, bytes) in arrays {
        zip.start_file(format!("{name}.npy"), options).map_err(|e| e.to_string())?;
        zip.write_all(bytes).map_err(|e| e.to_string())?;
    }
    zip.finish().map_err(|e| e.to_string())?;
    Ok(())
}

fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
    writer.write_image_data(rgba).map_err(|e| e.to_string())?;
    Ok(())
}

fn safe(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(zone), Some(out)) = (args.next(), args.next()) else {
        fatal("usage: clean_export <zone> <out dir>   (IW4L_GAMES points at the games root)");
    };
    let out = PathBuf::from(out);
    std::fs::create_dir_all(out.join("images")).unwrap_or_else(|e| fatal(&e.to_string()));

    let games = asset_transport::games_root_from_env().unwrap_or_else(|e| fatal(&e));
    let found = asset_transport::find_zone_file(&games, &zone).unwrap_or_else(|e| fatal(&e));
    if asset_transport::zone_game_for_path(&found.path) != Some(asset_core::ZoneGame::Iw4) {
        fatal("only MW2 (IW4) zones are supported");
    }
    let common = asset_transport::find_runtime_common_mp(&games, &found.path).map(|z| z.path);
    let prepared = match bevy::tasks::futures_lite::future::block_on(assets::load_prepared_match(
        Ok(found.path),
        common,
        asset_transport::LoadProgress::default(),
    )) {
        assets::MatchLoadOutcome::Ready(prepared) => prepared,
        assets::MatchLoadOutcome::Canceled => fatal("map load canceled"),
    };

    let world = &prepared.world;
    let catalog = &prepared.materials.population;
    let map_ids = &prepared.materials.map_ids;

    // Materials: one row per catalog material that geometry uses.
    let mut rows: HashMap<usize, u32> = HashMap::new();
    let mut materials_json = Vec::new();
    let mut skipped: HashMap<&'static str, u32> = HashMap::new();
    let mut material_row = |local: Option<usize>, skipped: &mut HashMap<&'static str, u32>| -> Option<u32> {
        let canonical = map_ids.get(local?).copied().flatten()?;
        if let Some(&row) = rows.get(&canonical) {
            return Some(row);
        }
        let m = catalog.materials.get(canonical)?;
        let name = m.name.as_str().to_owned();
        if name.contains("sky") || m.technique_set.as_str().contains("sky") {
            *skipped.entry("sky").or_default() += 1;
            return None;
        }
        let alpha = match catalog.agreed_draw_mode(m) {
            Some(asset_material::MaterialDrawMode::AlphaTest { .. }) => 1,
            Some(asset_material::MaterialDrawMode::Opaque) | None => 0,
            Some(_) => 2,
        };
        let image = m
            .textures
            .iter()
            .find(|t| t.semantic == asset_material::TS_COLOR_MAP)
            .and_then(|t| t.image)
            .and_then(|i| catalog.images.get(i).map(|img| (i, img)))
            .and_then(|(i, img)| {
                let decoded = img.decoded.as_deref()?;
                let (w, h, rgba) = asset_material::decoded_image_top_level_rgba8(decoded).ok()?;
                let file = format!("images/{}-{i:05}.png", safe(img.name.as_str()));
                let path = out.join(&file);
                if !path.exists() {
                    write_png(&path, w, h, &rgba).ok()?;
                }
                Some(file)
            });
        if image.is_none() {
            *skipped.entry("no colour map").or_default() += 1;
        }
        materials_json.push(serde_json::json!({
            "name": name,
            "image": image,
            "alpha": alpha,
            "surface": 0,
        }));
        let row = materials_json.len() as u32;
        rows.insert(canonical, row);
        Some(row)
    };

    let mut render = Render::default();
    if let Some(draw) = world.draw.as_ref() {
        for batch in &draw.batches {
            if let Some(row) = material_row(batch.material, &mut skipped) {
                if !render.push(&batch.mesh, row, Mat3::IDENTITY, Vec3::ZERO, 1.0) {
                    *skipped.entry("world batch unreadable").or_default() += 1;
                }
            }
        }
    }
    let mut placements = 0;
    for placement in world.static_model_instances.iter().flatten() {
        let Some(model) = world.static_model_meshes.get(placement.mesh) else { continue };
        let basis = Mat3::from_cols(placement.axis[0].into(), placement.axis[1].into(), placement.axis[2].into());
        for surface in model.surfaces() {
            if let Some(row) = material_row(surface.material, &mut skipped) {
                render.push(&surface.mesh, row, basis, placement.origin.into(), placement.scale);
            }
        }
        placements += 1;
    }

    // Collision: player-solid brushes and the clip mesh, kept exactly.
    let mut triangles: Vec<[[f32; 3]; 3]> = Vec::new();
    let mut surfaces: Vec<u32> = Vec::new();
    let clip = prepared.clip.as_ref().unwrap_or_else(|| fatal("map has no clip collision"));
    let mut brush_count = 0;
    for brush in &clip.brushes {
        if brush.contents & MASK_PLAYER_SOLID == 0 {
            continue;
        }
        brush_count += 1;
        let mut faces = Vec::new();
        brush_triangles(&brush.planes, &mut faces);
        for (tri, plane) in faces {
            triangles.push(tri.map(position));
            surfaces.push(surface_type(brush.plane_surface_flags.get(plane).copied().unwrap_or(0)));
        }
    }
    let mesh = &clip.mesh;
    let mut mesh_tris = 0;
    for part in &mesh.partitions {
        let base = part.first_vert_segment as usize * 1024;
        let first = part.first_tri as usize;
        for ti in first..first + part.tri_count as usize {
            if mesh.tri_content_flags.get(ti).is_some_and(|c| c & MASK_PLAYER_SOLID == 0) {
                continue;
            }
            let corner = |k: usize| -> Option<[f32; 3]> {
                let i = *mesh.tri_indices.get(3 * ti + k)? as usize;
                mesh.verts.get(base + i).copied()
            };
            let (Some(a), Some(b), Some(c)) = (corner(0), corner(1), corner(2)) else { continue };
            triangles.push([a, b, c].map(position));
            surfaces.push(surface_type(mesh.tri_surface_flags.get(ti).copied().unwrap_or(0)));
            mesh_tris += 1;
        }
    }

    // Spawns: team deathmatch first, then any other.
    let mut spawns: Vec<&asset_world::SpawnPoint> = prepared.prepared_map.spawns.iter().collect();
    spawns.sort_by_key(|s| (!s.classname.contains("tdm"), s.classname.clone()));
    let spawns_json: Vec<_> = spawns
        .iter()
        .map(|s| {
            let p = position(s.origin);
            let yaw = s.angles[1].to_radians();
            // IW4 forward (cos y, sin y, 0) is engine (cos y, 0, -sin y); heading h faces (sin h, 0, cos h).
            let heading = yaw.cos().atan2(-yaw.sin());
            serde_json::json!([p[0], p[1], p[2], heading])
        })
        .collect();

    write_npz(&out.join("render.npz"), &[
        ("positions", npy("<f4", &[render.positions.len(), 3], &f32_bytes(&render.positions))),
        ("normals", npy("<f4", &[render.normals.len(), 3], &f32_bytes(&render.normals))),
        ("uv", npy("<f4", &[render.uv.len(), 2], &f32_bytes(&render.uv))),
        ("material", npy("<u4", &[render.material.len()], &u32_bytes(&render.material))),
        ("indices", npy("<u4", &[render.indices.len()], &u32_bytes(&render.indices))),
    ])
    .unwrap_or_else(|e| fatal(&e));
    let flat: Vec<[f32; 3]> = triangles.iter().flatten().copied().collect();
    write_npz(&out.join("collision.npz"), &[
        ("triangles", npy("<f4", &[triangles.len(), 3, 3], &f32_bytes(&flat))),
        ("surface", npy("<u4", &[surfaces.len()], &u32_bytes(&surfaces))),
    ])
    .unwrap_or_else(|e| fatal(&e));
    let title = zone.strip_prefix("mp_").unwrap_or(&zone);
    let title = title[..1].to_uppercase() + &title[1..];
    let export = serde_json::json!({
        "zone": zone,
        "title": title,
        "surface_names": SURFACE_NAMES,
        "spawns": spawns_json,
        "materials": materials_json,
    });
    std::fs::write(out.join("export.json"), serde_json::to_string_pretty(&export).unwrap())
        .unwrap_or_else(|e| fatal(&e.to_string()));
    println!(
        "clean_export {zone}: {} materials, {} render triangles ({placements} static models), \
         {} collision triangles ({brush_count} brushes, {mesh_tris} mesh), {} spawns, skipped {skipped:?} -> {} (PRIVATE)",
        rows.len(),
        render.indices.len() / 3,
        triangles.len(),
        spawns_json.len(),
        out.display()
    );
}
