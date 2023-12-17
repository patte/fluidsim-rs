use bevy::prelude::*;

use bevy::window::PrimaryWindow;

use bevy_inspector_egui::{
    bevy_egui::EguiContext,
    egui::{self, FontId, RichText},
};

use super::Config;
use super::Measurements;

pub fn inspector_ui(
    world: &mut World,
    mut disabled: Local<bool>,
    mut last_time: Local<f32>,
    mut last_delta_t: Local<f32>,
) {
    let space_pressed = world.resource::<Input<KeyCode>>().just_pressed(KeyCode::H);
    if space_pressed {
        *disabled = !*disabled;
    }
    if *disabled {
        return;
    }

    let time = world.resource::<Time>().clone();
    if time.elapsed_seconds() - *last_time > 2. {
        *last_delta_t = time.delta_seconds();
        *last_time = time.elapsed_seconds();
    }
    let tps_local = if *last_delta_t > 0. {
        1. / *last_delta_t
    } else {
        0.
    };

    let measurements = world.resource::<Measurements>().clone();
    let delta_t = measurements.delta_t;
    let tps = measurements.tps;
    let p0_position = measurements.p0_position;
    let p0_predicted_position = measurements.p0_predicted_position;
    let p0_velocity = measurements.p0_velocity;
    let p0_density = measurements.p0_density;
    let p0_max_density_far = measurements.p0_max_density_far;

    let mut egui_context = world
        .query_filtered::<&mut EguiContext, With<PrimaryWindow>>()
        .single(world)
        .clone();

    let config = world.resource::<Config>().clone();

    egui::Window::new("Config")
        .default_width(50.)
        .default_height(600.)
        .default_open(false)
        .show(egui_context.get_mut(), |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                bevy_inspector_egui::bevy_inspector::ui_for_resource::<Config>(world, ui);

                ui.separator();
                ui.label(
                    RichText::new(format!(
                        "{}",
                        if tps > 300. {
                            "🚀"
                        } else if tps > 199. {
                            "👌"
                        } else if tps > 120. {
                            "🐢"
                        } else {
                            "🐌"
                        }
                    ))
                    .font(FontId::proportional(40.0)),
                );
                ui.label(format!("delta_t: {:.6} tps: {:.1}", delta_t, tps,));
                ui.label(format!(
                    "delta_t: {:.6} tps: {:.1}",
                    *last_delta_t, tps_local,
                ));
                ui.separator();
                if config.mark_sample_particle_neighbors {
                    ui.label(format!("position     : {:?}", p0_position));
                    ui.label(format!("pred position: {:?}", p0_predicted_position));
                    ui.label(format!("velocity     : {:?}", p0_velocity));
                    ui.label(format!("density      : {:?}", p0_density.far));
                    ui.label(format!("density near : {:?}", p0_density.near));
                    ui.label(format!(
                        "max density : {:?}",
                        p0_max_density_far.to_string()
                    ));
                    ui.separator();
                }
                ui.label("h: toggle ui");
                ui.label("i: reset config");
                ui.label("u: load config");
                ui.label("z: save config");
                ui.label("n: spawn 1 particle");
                ui.label("m: spawn 10 particles");
                ui.label("space: reset positions");
            });
        });
}
