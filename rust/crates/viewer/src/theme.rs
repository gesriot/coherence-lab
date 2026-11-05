use eframe::egui::{
    self, Color32, FontData, FontDefinitions, FontFamily, FontId, RichText, Stroke, TextStyle, Ui,
};

pub(crate) const BG: Color32 = Color32::from_rgb(16, 20, 16);
pub(crate) const SURFACE: Color32 = Color32::from_rgb(24, 32, 27);
pub(crate) const RAISED: Color32 = Color32::from_rgb(34, 44, 38);
pub(crate) const FG: Color32 = Color32::from_rgb(235, 230, 218);
pub(crate) const MUTED: Color32 = Color32::from_rgb(154, 163, 150);
pub(crate) const LINE: Color32 = Color32::from_rgb(49, 64, 56);
pub(crate) const ACCENT: Color32 = Color32::from_rgb(224, 161, 90);
pub(crate) const ACCENT_INK: Color32 = Color32::from_rgb(36, 24, 12);

pub(crate) fn display(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name("display".into()))
}
pub(crate) fn setup(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    for (name, data) in [
        (
            "Outfit",
            include_bytes!("../fonts/Outfit-Regular.ttf").as_slice(),
        ),
        (
            "Newsreader",
            include_bytes!("../fonts/Newsreader-Regular.ttf").as_slice(),
        ),
        (
            "IBM Plex Mono",
            include_bytes!("../fonts/IBMPlexMono-Regular.ttf").as_slice(),
        ),
    ] {
        fonts
            .font_data
            .insert(name.into(), FontData::from_static(data).into());
    }
    let mut heading = vec!["Newsreader".into()];
    heading.extend(fonts.families[&FontFamily::Proportional].clone());
    fonts
        .families
        .insert(FontFamily::Name("display".into()), heading);
    fonts
        .families
        .entry(FontFamily::Proportional)
        .or_default()
        .insert(0, "Outfit".into());
    fonts
        .families
        .entry(FontFamily::Monospace)
        .or_default()
        .insert(0, "IBM Plex Mono".into());
    ctx.set_fonts(fonts);
    let mut style = egui::Style {
        visuals: egui::Visuals::dark(),
        ..Default::default()
    };
    style.visuals.override_text_color = Some(FG);
    style.visuals.panel_fill = BG;
    style.visuals.window_fill = SURFACE;
    style.visuals.extreme_bg_color = BG;
    style.visuals.faint_bg_color = RAISED;
    style.visuals.selection.bg_fill = ACCENT;
    style.visuals.selection.stroke = Stroke::new(1., ACCENT_INK);
    style.visuals.widgets.noninteractive.bg_stroke = Stroke::new(1., LINE);
    for (active, widget) in [
        (false, &mut style.visuals.widgets.inactive),
        (true, &mut style.visuals.widgets.hovered),
        (true, &mut style.visuals.widgets.active),
    ] {
        widget.bg_fill = BG;
        widget.weak_bg_fill = BG;
        widget.bg_stroke = Stroke::new(1., if active { ACCENT } else { LINE });
        widget.fg_stroke = Stroke::new(1., FG);
        widget.corner_radius = 22.into();
    }
    style.spacing.item_spacing = egui::vec2(8., 8.);
    style.spacing.button_padding = egui::vec2(14., 8.);
    style
        .text_styles
        .insert(TextStyle::Body, FontId::proportional(14.));
    style
        .text_styles
        .insert(TextStyle::Button, FontId::proportional(14.));
    style
        .text_styles
        .insert(TextStyle::Small, FontId::proportional(12.));
    style
        .text_styles
        .insert(TextStyle::Monospace, FontId::monospace(12.));
    style.text_styles.insert(TextStyle::Heading, display(24.));
    ctx.set_theme(egui::Theme::Dark);
    ctx.set_style_of(egui::Theme::Dark, style);
}
pub(crate) fn card(ui: &mut Ui, body: impl FnOnce(&mut Ui)) {
    egui::Frame::new()
        .fill(SURFACE)
        .stroke(Stroke::new(1., LINE))
        .corner_radius(16)
        .inner_margin(16)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            body(ui);
        });
}
pub(crate) fn heading(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).font(display(24.)).color(FG));
}
pub(crate) fn muted(ui: &mut Ui, text: impl Into<String>) {
    ui.add(egui::Label::new(RichText::new(text).color(MUTED)).wrap());
}
pub(crate) fn small(ui: &mut Ui, text: impl Into<String>) {
    ui.add(egui::Label::new(RichText::new(text).size(12.).color(MUTED)).wrap());
}
pub(crate) fn mono(ui: &mut Ui, text: impl Into<String>) {
    ui.add(egui::Label::new(RichText::new(text).monospace().size(12.).color(MUTED)).wrap());
}
pub(crate) fn pill(ui: &mut Ui, label: &str, on: bool, enabled: bool) -> bool {
    ui.add_enabled(
        enabled,
        egui::Button::new(RichText::new(label).color(if on { ACCENT_INK } else { FG }))
            .fill(if on { ACCENT } else { BG })
            .stroke(Stroke::new(1., if on { ACCENT } else { LINE }))
            .corner_radius(24)
            .min_size(egui::vec2(0., 44.)),
    )
    .clicked()
}
pub(crate) fn bar(ui: &mut Ui, value: f64) {
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 6.), egui::Sense::hover());
    ui.painter().rect_filled(rect, 3, BG);
    let width = rect.width() * (jscompat::round(value.clamp(0., 1.) * 100.) / 100.) as f32;
    ui.painter().rect_filled(
        egui::Rect::from_min_size(rect.min, egui::vec2(width, 6.)),
        3,
        ACCENT,
    );
}
