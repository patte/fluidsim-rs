use bevy::prelude::*;

use crate::{Config, Particle, Velocity};

pub fn move_system(
    config: Res<Config>,
    time: Res<Time>,
    mut particles_query: Query<(&mut Transform, &Velocity), With<Particle>>,
) {
    if config.is_paused {
        return;
    }
    let delta_t = time.delta_seconds() * config.time_scale;

    particles_query
        .par_iter_mut()
        .for_each(|(mut transform, velocity)| {
            transform.translation += velocity.0.extend(0.) * (delta_t);
        });
}
