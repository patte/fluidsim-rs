use bevy::prelude::*;

use crate::{Config, InstanceMaterialData, Particle};

pub fn move_system(
    config: Res<Config>,
    time: Res<Time>,
    mut particles_query: Query<&mut InstanceMaterialData, With<Particle>>,
) {
    if config.is_paused {
        return;
    }
    let delta_t = time.delta_seconds() * config.time_scale;

    particles_query.par_iter_mut().for_each(|mut data| {
        for instance in data.iter_mut() {
            instance.position += instance.velocity.extend(0.) * delta_t;
        }
    });
}
