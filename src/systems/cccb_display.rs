use crate::Config;
use bevy::prelude::*;
use bevy::render::view::screenshot::ScreenshotManager;
use bevy::window::PrimaryWindow;
use cccb_display::{CccbDisplayImagePackage, CccbImageSender};

pub fn cccb_display_system(
    main_window: Query<Entity, With<PrimaryWindow>>,
    mut screenshot_manager: ResMut<ScreenshotManager>,
    mut elapsed: Local<f32>,
    time: Res<Time>,
    config: Res<Config>,
) {
    if !config.cccb_display_sender {
        return;
    }

    *elapsed += time.delta_seconds();

    // max pps 200
    if *elapsed < 1. / 200. {
        return;
    }
    *elapsed = 0.;

    let _ = screenshot_manager.take_screenshot(main_window.single(), move |img| {
        match img.try_into_dynamic() {
            Ok(dynamic_img) => {
                if dynamic_img.width() == 0 || dynamic_img.height() == 0 {
                    println!(
                        "Screenshot empty: {}x{}",
                        dynamic_img.width(),
                        dynamic_img.height()
                    );
                    return;
                }

                fn pix_is_on(pix: &image::Rgba<u8>) -> bool {
                    pix[0] + pix[1] + pix[2] > (60 * 3)
                }
                let img_packed = CccbDisplayImagePackage::new(dynamic_img, pix_is_on, true);

                // save screenshot to disk
                img_packed
                    .to_luma8()
                    .save("./screenshots/screenshot.png")
                    .unwrap();

                let mut sender = CccbImageSender::new_from_env();
                sender.send_package(&img_packed);
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    });
}
