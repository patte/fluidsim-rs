use bevy::prelude::*;
use rand::{thread_rng, Rng};

use super::Config;

pub fn get_random_transform(config: &Config) -> Transform {
    let width_half = config.bounding_box.width / 2.;
    let height_half = config.bounding_box.height / 2.;

    let mut rng = thread_rng();
    let x = rng.gen_range(-width_half..width_half);
    let y = rng.gen_range(-height_half..height_half);
    Transform::from_translation(Vec3::new(x, y, 0.))
}

pub fn get_position_in_grid(config: &Config, i: usize) -> Transform {
    let num_particles_per_row = (config.num_particles as f32).sqrt() as usize;

    let mut gap_between_particles = config.smoothing_radius * 0.9;

    let mut width_particles = num_particles_per_row as f32 * gap_between_particles;

    if width_particles > config.bounding_box.height.max(config.bounding_box.width) {
        println!("Warning: particles are too big for the bounding box");
        gap_between_particles = config.bounding_box.height.max(config.bounding_box.width)
            / num_particles_per_row as f32;
        width_particles = num_particles_per_row as f32 * gap_between_particles;
    }

    let start_x = -width_particles / 2.;
    let start_y = -width_particles / 2.;

    Transform::from_translation(Vec3::new(
        start_x + (i % num_particles_per_row) as f32 * gap_between_particles,
        start_y + (i / num_particles_per_row) as f32 * gap_between_particles,
        0.,
    ))
}

pub fn new_circle(radius: f32) -> Mesh {
    Circle {
        radius,
        ..default()
    }
    .mesh()
    .resolution(8)
    .into()
}
