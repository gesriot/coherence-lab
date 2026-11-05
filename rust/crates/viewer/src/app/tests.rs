use super::*;
use crate::BenchState;
use eframe::egui::{self, FontId, PointerButton, accesskit::Role};
use egui_kittest::{
    Harness,
    kittest::{NodeT, Queryable},
};
use lab::{Config, Episode, Policy, Scenario};
use std::{path::Path, sync::LazyLock};

fn harness(width: f32) -> Harness<'static, Viewer> {
    Harness::builder()
        .with_size(egui::vec2(width, 1000.))
        .with_step_dt(0.)
        .with_render_cursor(false)
        .build_eframe(|cc| Viewer::waiting(&cc.egui_ctx))
}
fn click(h: &mut Harness<'_, Viewer>, label: &str) {
    h.get_by_role_and_label(Role::Button, label)
        .click_accesskit();
    h.run();
}
fn snapshot(state: &State) -> Vec<u8> {
    serde_json::to_vec(&state.view()).expect("snapshot serializes")
}
#[test]
fn labelled_buttons_apply_episode_actions_without_resetting_on_mode_changes() {
    let mut h = harness(1440.);
    h.run();
    click(&mut h, "Пауза");
    assert!(!h.state().state.running);
    assert_eq!(h.state().state.view().step, 0);
    click(&mut h, "Пуск");
    assert!(h.state().state.running);
    click(&mut h, "Шаг");
    assert!(!h.state().state.running);
    assert_eq!(h.state().state.view().step, 1);
    let mut expected = Episode::new(Scenario::Continuity, 4, Policy::Full);
    expected.step(&Config::default());
    assert_eq!(
        snapshot(&h.state().state),
        serde_json::to_vec(&expected.snapshot(Policy::Full)).expect("snapshot")
    );
    let previous = snapshot(&h.state().state);
    click(&mut h, "Первый порядок");
    assert_eq!(snapshot(&h.state().state), previous);
    click(&mut h, "4×");
    assert_eq!(h.state().state.speed, Speed::Four);
    click(&mut h, "2×");
    assert_eq!(h.state().state.speed, Speed::Two);
    click(&mut h, "1×");
    assert_eq!(h.state().state.speed, Speed::One);
    click(&mut h, "Сначала");
    assert_eq!(h.state().state.view().step, 0);
    assert!(!h.state().state.running);
    click(&mut h, "Конфликт");
    assert_eq!(h.state().state.scenario, Scenario::Conflict);
    assert_eq!(
        snapshot(&h.state().state),
        serde_json::to_vec(
            &Episode::new(Scenario::Conflict, 4, Policy::FirstOrder).snapshot(Policy::FirstOrder)
        )
        .expect("snapshot")
    );
}
#[test]
fn confidence_button_is_disabled_outside_schema_modes_and_restores_its_value() {
    let mut h = harness(1440.);
    h.run();
    click(&mut h, "Подменить уверенность");
    assert!(h.state().state.inject);
    for (policy, label) in labels::POLICIES {
        click(&mut h, label);
        let enabled = matches!(policy, Policy::Full | Policy::RecordOnly);
        assert_eq!(
            !h.get_by_role_and_label(Role::Button, "Подменить уверенность")
                .accesskit_node()
                .is_disabled(),
            enabled
        );
        if !enabled {
            click(&mut h, "Подменить уверенность");
            assert!(h.state().state.inject);
        }
    }
}
#[test]
fn scene_pointer_uses_world_coordinates_and_ignores_padding() {
    let mut h = harness(1440.);
    h.run();
    scroll_scene(&mut h);
    let map = scene::Map::new(h.state().scene_rect);
    let point = map.point(30., 70.);
    for pos in [h.state().scene_rect.min + egui::vec2(1., 1.), point] {
        h.event(egui::Event::PointerMoved(pos));
        for pressed in [true, false] {
            h.event(egui::Event::PointerButton {
                pos,
                button: PointerButton::Primary,
                pressed,
                modifiers: Default::default(),
            });
        }
        h.run();
        if pos == point {
            assert!(h.state().state.pin_gaze);
            assert!((h.state().state.pin[0] - 30.).abs() < 0.001);
            assert!((h.state().state.pin[1] - 70.).abs() < 0.001);
        } else {
            assert!(!h.state().state.pin_gaze);
        }
    }
}
#[test]
fn pending_and_failed_protocol_use_reference_labels() {
    let mut h = harness(480.);
    h.run();
    assert!(!h.get_all_by_label("Идёт расчёт…").next().is_none());
    h.state_mut().state.bench = BenchState::Failed;
    h.run();
    assert!(
        !h.get_all_by_label("Не удалось рассчитать бенчмарк.")
            .next()
            .is_none()
    );
}
#[test]
fn each_embedded_face_alone_covers_cyrillic_and_protocol_symbols() {
    for (name, data) in crate::theme::FONTS {
        // Load the face without egui fallbacks, so a missing glyph cannot be borrowed.
        let ctx = egui::Context::default();
        let mut fonts = egui::FontDefinitions::empty();
        fonts
            .font_data
            .insert(name.into(), egui::FontData::from_static(data).into());
        fonts
            .families
            .insert(egui::FontFamily::Proportional, vec![name.into()]);
        ctx.set_fonts(fonts);
        // No renderer consumes the font atlas upload here.
        ctx.run_ui(egui::RawInput::default(), |_| {})
            .textures_delta
            .clear();
        ctx.fonts_mut(|fonts| {
            assert!(
                fonts.has_glyphs(
                    &FontId::proportional(14.),
                    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя–·°≥…×"
                ),
                "{name}"
            );
        });
    }
}

static REPORT: LazyLock<lab::BenchReport> = LazyLock::new(lab::run_bench);
#[test]
fn completed_protocol_displays_every_result_and_can_render_review_frames() -> anyhow::Result<()> {
    let mut h = harness(1440.);
    h.state_mut().state.bench = BenchState::Ready(REPORT.clone());
    h.run();
    for row in &REPORT.rows {
        assert!(
            !h.get_all_by_label(lab::format_rate(row.retention).as_str())
                .next()
                .is_none()
        );
        assert!(
            !h.get_all_by_label(lab::format_rate(row.pair).as_str())
                .next()
                .is_none()
        );
        assert!(!h.get_all_by_label(row.detail).next().is_none());
    }
    assert!(
        !h.get_all_by_label(lab::causal_label(REPORT.causal))
            .next()
            .is_none()
    );
    assert!(h.query_by_label("Идёт расчёт…").is_none());
    if let Some(path) = std::env::var_os("VIEWER_SCREENSHOTS") {
        render_frames(Path::new(&path))?;
    }
    Ok(())
}
fn save(h: &mut Harness<'_, Viewer>, path: &Path) -> anyhow::Result<()> {
    use anyhow::Context;
    h.run();
    let image = h
        .render()
        .map_err(anyhow::Error::msg)
        .context("render window")?;
    image.save(path).context("save window image")
}
fn scroll_scene(h: &mut Harness<'_, Viewer>) {
    let title = if h.state().state.show_truth {
        "Мир оценщика"
    } else {
        "Вход агента"
    };
    if let Some(node) = h
        .get_all_by_label(title)
        .max_by(|a, b| a.rect().height().total_cmp(&b.rect().height()))
    {
        node.scroll_to_me();
    }
    h.run();
}
fn render_frames(root: &Path) -> anyhow::Result<()> {
    use anyhow::{Context, bail};
    std::fs::create_dir_all(root).context("create screenshot directory")?;
    let root = root
        .canonicalize()
        .context("resolve screenshot directory")?;
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .context("resolve repository")?;
    if root.starts_with(repository) {
        bail!("screenshots must be outside the repository");
    }
    for width in [1440, 480] {
        for case in [
            "initial",
            "continuity-truth",
            "continuity-input",
            "conflict",
            "pinned",
            "protocol",
        ] {
            let mut h = harness(width as f32);
            {
                let s = &mut h.state_mut().state;
                if case == "conflict" {
                    s.act(Action::Scenario(Scenario::Conflict), Duration::ZERO);
                    s.act(Action::Seed("2".into()), Duration::ZERO);
                }
                let steps = match case {
                    "continuity-truth" | "continuity-input" => 30,
                    "conflict" => 40,
                    _ => 0,
                };
                for _ in 0..steps {
                    s.act(Action::Step, Duration::ZERO);
                }
                if case == "continuity-input" {
                    s.act(Action::Truth, Duration::ZERO);
                }
                if case == "pinned" {
                    s.act(Action::Pick(30., 70.), Duration::ZERO);
                    s.act(Action::Step, Duration::ZERO);
                    s.act(Action::Step, Duration::ZERO);
                }
                if case == "protocol" {
                    s.bench = BenchState::Ready(REPORT.clone());
                }
            }
            h.run();
            if case == "protocol" {
                h.get_by_label("Протокол").scroll_to_me();
                h.run();
                let shift = h.get_by_label("Протокол").rect().top() - 20.;
                h.event(egui::Event::PointerMoved(egui::pos2(8., 500.)));
                for (phase, delta) in [
                    (egui::TouchPhase::Start, 0.),
                    (egui::TouchPhase::Move, -shift),
                ] {
                    h.event(egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Point,
                        phase,
                        source: egui::MouseWheelSource::Trackpad,
                        delta: egui::vec2(0., delta),
                        modifiers: Default::default(),
                    });
                }
                h.run_steps(1);
                h.event(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    phase: egui::TouchPhase::End,
                    source: egui::MouseWheelSource::Trackpad,
                    delta: egui::Vec2::ZERO,
                    modifiers: Default::default(),
                });
                h.run();
            } else if case != "initial" {
                scroll_scene(&mut h);
            }
            save(&mut h, &root.join(format!("{width}x1000-{case}.png")))?;
        }
    }
    Ok(())
}
