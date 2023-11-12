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

    let mut egui_context = world
        .query_filtered::<&mut EguiContext, With<PrimaryWindow>>()
        .single(world)
        .clone();

    egui::Window::new("Config")
        .default_width(50.)
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
