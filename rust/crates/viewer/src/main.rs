use anyhow::{Context, Result};
fn main() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Когерентность")
            .with_inner_size([1440., 1000.])
            .with_min_inner_size([360., 480.]),
        ..Default::default()
    };
    eframe::run_native(
        "Когерентность",
        options,
        Box::new(|cc| Ok(Box::new(viewer::Viewer::new(&cc.egui_ctx)))),
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))
    .context("open laboratory window")
}
