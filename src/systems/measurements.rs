use bevy::prelude::*;

use crate::{Config, Density, Measurements, Particle, PredictedPosition, SpatialHash, Velocity};

pub fn measurements_system(
    time: Res<Time>,
    mut measurements: ResMut<Measurements>,
    config: Res<Config>,
    particles_query: Query<(&Transform, &PredictedPosition, &Velocity, &Density), With<Particle>>,
    spatial_hash: Res<SpatialHash>,
) {
    if config.mark_sample_particle_neighbors && spatial_hash.first_entity_id != Entity::from_raw(0)
    {
        let (p0_position, p0_predicted_position, p0_velocity, p0_density) =
            particles_query.get(spatial_hash.first_entity_id).unwrap();
        measurements.p0_position = p0_position.translation.clone();
        measurements.p0_predicted_position = p0_predicted_position.0.clone();
        measurements.p0_velocity = p0_velocity.0.clone();
        measurements.p0_density = p0_density.clone();
        if p0_density.far > measurements.p0_max_density_far {
            measurements.p0_max_density_far = p0_density.far;
        }
    }

    if config.is_paused {
        return;
    }

    measurements.delta_t = time.delta_seconds() * config.time_scale;
    measurements.tps = 1. / measurements.delta_t;
}
