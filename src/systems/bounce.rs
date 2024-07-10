use bevy::prelude::*;

use crate::{Config, InstanceMaterialData, Particle, SCALE_FACTOR2};

pub fn bounce_system(
    mut config: ResMut<Config>,
    mut particles_query: Query<&mut InstanceMaterialData, With<Particle>>,
) {
    if config.is_paused {
        return;
    }

    let width = config.bounding_box.width * SCALE_FACTOR2;
    let height = config.bounding_box.height * SCALE_FACTOR2;

    let half_size = Vec3::new(width / 2., height / 2., 0.0);

    particles_query.par_iter_mut().for_each(|mut data| {
        for instance in data.iter_mut() {
            let edge_dst = half_size - instance.position.abs();

            if edge_dst.x <= 0. {
                // switch direction
                if instance.velocity.x.signum() == instance.position.x.signum() {
                    instance.velocity.x *= -1. * (1.0 - config.damping);
                }

                // move inside
                instance.position.x += -instance.position.x.signum() * edge_dst.x.abs();
            }
            if edge_dst.y <= 0. {
                // switch direction
                if instance.velocity.y.signum() == instance.position.y.signum() {
                    instance.velocity.y = -instance.velocity.y * (1.0 - config.damping);
                }

                // move inside
                instance.position.y -= instance.position.y.signum() * edge_dst.y.abs();
            }
        }
    });

    if config.pause_after_next_frame {
        config.is_paused = true;
        config.pause_after_next_frame = false;
    }
}
