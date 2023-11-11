use bevy::prelude::*;
use rand::{thread_rng, Rng};

use super::RADIUS;

pub fn get_random_transform(window: &Window) -> Transform {
    let width_half = window.width() / 2.;
    let height_half = window.height() / 2.;

    let mut rng = thread_rng();
    let x = rng.gen_range(-width_half..width_half);
    let y = rng.gen_range(-height_half..height_half);
    Transform::from_translation(Vec3::new(x, y, 0.))
}

pub fn new_circle() -> shape::Circle {
    shape::Circle {
        radius: RADIUS,
        vertices: 4,
        ..default()
    }
}
