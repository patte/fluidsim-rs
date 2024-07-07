use bevy::prelude::*;

use crate::{Config, Particle, Velocity, SCALE_FACTOR2};

pub fn bounce_system(
    mut config: ResMut<Config>,
    mut particles_query: Query<(&mut Transform, &mut Velocity), With<Particle>>,
) {
    if config.is_paused {
        return;
    }

    let width = config.bounding_box.width * SCALE_FACTOR2;
    let height = config.bounding_box.height * SCALE_FACTOR2;

    let half_size = Vec3::new(width / 2., height / 2., 0.0);

    particles_query
        .par_iter_mut()
        .for_each(|(mut transform, mut velocity)| {
            let edge_dst = half_size - transform.translation.abs();

            if edge_dst.x <= 0. {
                // switch direction
                if velocity.0.x.signum() == transform.translation.x.signum() {
                    velocity.0.x *= -1. * (1.0 - config.damping);
                }

                // move inside
                transform.translation.x += -transform.translation.x.signum() * edge_dst.x.abs();
            }
            if edge_dst.y <= 0. {
                // switch direction
                if velocity.0.y.signum() == transform.translation.y.signum() {
                    velocity.0.y = -velocity.0.y * (1.0 - config.damping);
                }

                // move inside
                transform.translation.y -= transform.translation.y.signum() * edge_dst.y.abs();
            }
        });

    if config.pause_after_next_frame {
        config.is_paused = true;
        config.pause_after_next_frame = false;
    }
}
