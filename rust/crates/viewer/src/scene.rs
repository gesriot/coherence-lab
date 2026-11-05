use crate::theme::*;
use eframe::egui::{
    self, Align2, Color32, FontId, Painter, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, Ui,
};
use lab::{Shape as BodyShape, Snapshot};

#[derive(Clone, Copy)]
pub(crate) struct Map {
    origin: Pos2,
    scale: f32,
}
impl Map {
    pub(crate) fn new(rect: Rect) -> Self {
        let scale = ((rect.width() - 36.).min(rect.height() - 36.) / 100.).max(0.);
        Self {
            origin: rect.center() - egui::vec2(50. * scale, 50. * scale),
            scale,
        }
    }
    pub(crate) fn point(self, x: f64, y: f64) -> Pos2 {
        self.origin + egui::vec2(x as f32 * self.scale, y as f32 * self.scale)
    }
    pub(crate) fn pick(self, pos: Pos2) -> Option<[f64; 2]> {
        if self.scale <= 0. {
            return None;
        }
        let p = (pos - self.origin) / self.scale;
        ((0. ..=100.).contains(&p.x) && (0. ..=100.).contains(&p.y))
            .then_some([f64::from(p.x), f64::from(p.y)])
    }
}
pub(crate) fn hsl(h: f64, s: f64, l: f64) -> Color32 {
    let h = h.rem_euclid(360.) / 30.;
    let s = s.clamp(0., 100.) / 100.;
    let l = l.clamp(0., 100.) / 100.;
    let a = s * l.min(1. - l);
    let channel = |n: f64| {
        let k = (n + h) % 12.;
        ((l - a * (k - 3.).min(9. - k).clamp(-1., 1.)) * 255.).round() as u8
    };
    Color32::from_rgb(channel(0.), channel(8.), channel(4.))
}
fn dashed_circle(p: &Painter, center: Pos2, radius: f32, stroke: Stroke, dash: f32, gap: f32) {
    let count = (radius * 2.).ceil().max(32.) as usize;
    let points: Vec<_> = (0..=count)
        .map(|i| {
            let a = i as f32 / count as f32 * std::f32::consts::TAU;
            center + egui::vec2(a.cos(), a.sin()) * radius
        })
        .collect();
    p.extend(Shape::dashed_line(&points, stroke, dash, gap));
}
pub(crate) fn draw(
    ui: &mut Ui,
    view: &Snapshot<'_>,
    truth: bool,
    hypotheses: bool,
) -> (Rect, Option<[f64; 2]>) {
    let side = ui.available_width();
    let (rect, response) = ui.allocate_exact_size(egui::vec2(side, side), Sense::click());
    response.widget_info(|| {
        egui::WidgetInfo::labeled(
            egui::Role::Image,
            true,
            if truth {
                "Мир оценщика"
            } else {
                "Вход агента"
            },
        )
    });
    let p = ui.painter_at(rect);
    let m = Map::new(rect);
    p.rect_filled(rect, 0, BG);
    for g in (0..=100).step_by(20) {
        let g = f64::from(g);
        let stroke = Stroke::new(1., LINE.gamma_multiply(0.7));
        p.line_segment([m.point(g, 0.), m.point(g, 100.)], stroke);
        p.line_segment([m.point(0., g), m.point(100., g)], stroke);
    }
    if truth {
        let wall = view.occluder;
        let r = Rect::from_min_size(
            m.point(wall.x, wall.y),
            egui::vec2(wall.w as f32 * m.scale, wall.h as f32 * m.scale),
        );
        p.rect_filled(r, 0, MUTED.gamma_multiply(0.28));
        let clipped = p.with_clip_rect(r.intersect(rect));
        let span = r.width() + r.height();
        let mut i = -span;
        while i < span {
            clipped.line_segment(
                [
                    r.min + egui::vec2(i, 0.),
                    r.min + egui::vec2(i + r.height(), r.height()),
                ],
                Stroke::new(1., FG.gamma_multiply(0.55)),
            );
            i += 8.;
        }
        p.rect_stroke(
            r.translate(egui::vec2(0.5, 0.5)),
            0,
            Stroke::new(1., ACCENT),
            StrokeKind::Middle,
        );
        for body in view.objects {
            let report = view.evaluator.objects.iter().find(|r| r.id == body.id);
            if report.is_some_and(|r| r.vanished) {
                continue;
            }
            let alpha = if report.is_some_and(|r| r.occluded) {
                0.35
            } else {
                1.
            };
            let center = m.point(body.x, body.y);
            let radius = body.r as f32 * m.scale;
            let fill = hsl(body.hue, 68., 52.).gamma_multiply(alpha);
            let stroke = Stroke::new(1.25, FG.gamma_multiply(alpha));
            match body.shape {
                BodyShape::Disk => {
                    p.circle(center, radius, fill, stroke);
                }
                BodyShape::Square => {
                    p.rect(
                        Rect::from_center_size(center, egui::vec2(2. * radius, 2. * radius)),
                        0,
                        fill,
                        stroke,
                        StrokeKind::Middle,
                    );
                }
                _ => {
                    p.add(Shape::convex_polygon(
                        vec![
                            center + egui::vec2(0., -radius),
                            center + egui::vec2(radius, radius * 0.75),
                            center + egui::vec2(-radius, radius * 0.75),
                        ],
                        fill,
                        stroke,
                    ));
                }
            }
            p.text(
                center + egui::vec2(radius + 4., -2.),
                Align2::LEFT_BOTTOM,
                body.id,
                FontId::proportional(12.),
                FG,
            );
        }
    } else {
        let gaze = view.obs_gaze;
        let cell = gaze.aperture * 2. / view.grid as f64;
        for (i, sample) in view.cells.iter().enumerate() {
            let pos = m.point(
                gaze.x - gaze.aperture + (i % view.grid) as f64 * cell,
                gaze.y - gaze.aperture + (i / view.grid) as f64 * cell,
            );
            let size = (cell as f32 * m.scale).ceil();
            p.rect_filled(
                Rect::from_min_size(pos, egui::vec2(size, size)),
                0,
                hsl(sample.hue, sample.saturation, sample.lightness),
            );
        }
    }
    if hypotheses {
        for track in view.tracks {
            let alpha = (0.35 + track.confidence * 0.65) as f32;
            let center = m.point(track.x, track.y);
            dashed_circle(
                &p,
                center,
                5.2 * m.scale,
                Stroke::new(
                    if track.in_workspace { 1.75 } else { 1. },
                    hsl(track.hue, 70., 62.).gamma_multiply(alpha),
                ),
                4.,
                3.,
            );
            p.text(
                center + egui::vec2(8., -6.),
                Align2::LEFT_BOTTOM,
                format!("T{}", track.id),
                FontId::monospace(11.),
                FG.gamma_multiply(alpha),
            );
        }
    }
    let gaze = view.obs_gaze;
    let center = m.point(gaze.x, gaze.y);
    let size = (gaze.aperture * 2.) as f32 * m.scale;
    p.rect_stroke(
        Rect::from_center_size(center, egui::vec2(size, size)),
        0,
        Stroke::new(1.5, ACCENT),
        StrokeKind::Middle,
    );
    p.line_segment(
        [center - egui::vec2(6., 0.), center + egui::vec2(6., 0.)],
        Stroke::new(1.5, ACCENT),
    );
    p.line_segment(
        [center - egui::vec2(0., 6.), center + egui::vec2(0., 6.)],
        Stroke::new(1.5, ACCENT),
    );
    dashed_circle(
        &p,
        m.point(view.next_gaze.x, view.next_gaze.y),
        3.5,
        Stroke::new(1.5, MUTED),
        2.,
        3.,
    );
    (
        rect,
        response
            .clicked()
            .then(|| response.interact_pointer_pos())
            .flatten()
            .and_then(|pos| m.pick(pos)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn click_mapping_inverts_drawing_and_rejects_padding() {
        for size in [
            egui::vec2(300., 300.),
            egui::vec2(800., 800.),
            egui::vec2(600., 350.),
        ] {
            let rect = Rect::from_min_size(egui::pos2(42., 91.), size);
            let m = Map::new(rect);
            for [x, y] in [[0., 0.], [100., 100.], [30., 70.], [50., 50.]] {
                let got = m.pick(m.point(x, y)).expect("world point");
                assert!((got[0] - x).abs() < 0.00002 && (got[1] - y).abs() < 0.00002);
            }
            assert!(m.pick(rect.min).is_none());
            assert!(m.pick(m.point(-1., 50.)).is_none());
        }
    }
    #[test]
    fn css_hsl_wraps_hue_clamps_percentages_and_encodes_srgb() {
        assert_eq!(hsl(0., 100., 50.), Color32::RED);
        assert_eq!(hsl(-240., 100., 50.), Color32::GREEN);
        assert_eq!(hsl(600., 100., 50.), Color32::BLUE);
        assert_eq!(hsl(90., -1., 50.), Color32::from_gray(128));
        assert_eq!(hsl(90., 200., 200.), Color32::WHITE);
        assert_eq!(hsl(90., 200., -5.), Color32::BLACK);
    }
}
