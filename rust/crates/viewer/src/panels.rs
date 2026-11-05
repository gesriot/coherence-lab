use crate::{Action, State, labels, scene, theme::*};
use eframe::egui::{self, FontId, Rect, RichText, Sense, Stroke, Ui};
use lab::{Snapshot, decimal};

pub(crate) fn present(ui: &mut Ui, v: &Snapshot<'_>) {
    card(ui, |ui| {
        heading(ui, "Пузырь настоящего");
        mono(
            ui,
            format!(
                "{}/{} в сцене",
                v.tracks.iter().filter(|t| t.in_workspace).count(),
                v.workspace_capacity
            ),
        );
        if v.tracks.is_empty() {
            muted(
                ui,
                "Сцена пуста. Контур ещё не связал измерения в сущности.",
            );
        }
        for t in v.tracks {
            egui::Frame::new()
                .fill(RAISED)
                .stroke(Stroke::new(1., LINE))
                .corner_radius(8)
                .inner_margin(10)
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal_wrapped(|ui| {
                        let (r, _) = ui.allocate_exact_size(egui::vec2(12., 12.), Sense::hover());
                        ui.painter()
                            .circle_filled(r.center(), 6., scene::hsl(t.hue, 70., 52.));
                        ui.label(RichText::new(format!("T{}", t.id)).monospace());
                        small(ui, labels::source(t.source));
                        small(
                            ui,
                            if t.in_workspace {
                                "в сцене"
                            } else {
                                "периферия"
                            },
                        );
                    });
                    small(
                        ui,
                        format!(
                            "{}{}",
                            labels::reason(t.reason),
                            if t.conflict {
                                " · конфликт связывания"
                            } else {
                                ""
                            }
                        ),
                    );
                    bar(ui, 1. - t.uncertainty);
                    mono(
                        ui,
                        format!(
                            "увер. {} · неопр. {} · свидет. {}",
                            decimal(t.confidence, 2),
                            decimal(t.uncertainty, 2),
                            decimal(t.evidence, 2)
                        ),
                    );
                });
        }
    });
}
pub(crate) fn attention(ui: &mut Ui, v: &Snapshot<'_>) {
    card(ui, |ui| {
        heading(ui, "Модель внимания");
        let a = v.attention;
        if a.active {
            ui.label(a.note);
            muted(
                ui,
                format!(
                    "Намерение: {}{}{}",
                    labels::attention(a.reason),
                    a.focus_id
                        .map_or(String::new(), |id| format!(" · фокус T{id}")),
                    if a.influences_action {
                        ""
                    } else {
                        " · на действие не влияет"
                    }
                ),
            );
            mono(
                ui,
                format!(
                    "ошибка самомодели {} · память {} кадров",
                    decimal(a.self_error, 2),
                    v.memory_frames
                ),
            );
        } else {
            muted(
                ui,
                format!(
                    "{} {}",
                    a.note,
                    if v.memory_frames > 0 {
                        format!("Кадров в памяти: {}.", v.memory_frames)
                    } else {
                        String::new()
                    }
                ),
            );
        }
        ui.add_space(8.);
        small(ui, "Измерение, не сцена".to_uppercase());
        let (r, response) = ui.allocate_exact_size(egui::vec2(144., 144.), Sense::hover());
        response.widget_info(|| {
            egui::WidgetInfo::labeled(egui::Role::Image, true, "Сырые клетки сенсорного окна")
        });
        for (i, c) in v.cells.iter().enumerate() {
            let pos = r.min + egui::vec2((i % 12) as f32 * 12., (i / 12) as f32 * 12.);
            ui.painter().rect_filled(
                Rect::from_min_size(pos, egui::vec2(11., 11.)),
                0,
                scene::hsl(c.hue, c.saturation, c.lightness),
            );
        }
    });
}
pub(crate) fn coherence(ui: &mut Ui, v: &Snapshot<'_>) {
    card(ui, |ui| {
        heading(ui, "Когерентность");
        muted(
            ui,
            "Меньшее значение не означает «больше сознания»: это лишь ошибка согласования. Пустая сцена при непустом входе штрафуется.",
        );
        ui.label(
            RichText::new(format!("E {}", decimal(v.energy.total, 2)))
                .monospace()
                .color(ACCENT),
        );
        let e = v.energy;
        let rows = [
            ("Сенсорика", Some(e.sensory)),
            ("Связывание", Some(e.binding)),
            ("Время", Some(e.temporal)),
            ("Действие", Some(e.action)),
            ("Самомодель", e.self_error),
            ("Ресурс", Some(e.regularization)),
        ];
        let energy = |ui: &mut Ui, label: &str, value: Option<f64>| {
            ui.horizontal(|ui| {
                small(ui, label);
                mono(ui, decimal(value, 2));
            });
            bar(ui, value.unwrap_or(0.));
        };
        if ui.available_width() >= 590. {
            ui.columns(2, |cols| {
                for (i, (label, value)) in rows.into_iter().enumerate() {
                    energy(&mut cols[i % 2], label, value);
                }
            });
        } else {
            for (label, value) in rows {
                energy(ui, label, value);
            }
        }
        ui.add_space(8.);
        small(
            ui,
            "Итерации согласования · E от исходной сцены до остановки",
        );
        let (r, response) = ui.allocate_exact_size(egui::vec2(176., 48.), Sense::hover());
        let trace = v
            .energy_trace
            .iter()
            .map(|x| decimal(*x, 3))
            .collect::<Vec<_>>()
            .join(" → ");
        response.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Image, true, &trace));
        let points: Vec<_> = v
            .energy_trace
            .iter()
            .enumerate()
            .map(|(i, x)| {
                r.min
                    + egui::vec2(
                        (2. + 156. * i as f32 / (v.energy_trace.len() - 1).max(1) as f32) * 176.
                            / 160.,
                        (40. - 38. * (*x as f32)) * 48. / 42.,
                    )
            })
            .collect();
        if points.len() > 1 {
            ui.painter()
                .add(egui::Shape::line(points, Stroke::new(2., ACCENT)));
        }
        let (r, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 40.), Sense::hover());
        let n = v.history.len().max(1) as f32;
        let width = ((r.width() - 4. * (n - 1.)) / n).max(1.);
        for (i, h) in v.history.iter().enumerate() {
            let height = 40. * (jscompat::round(h.total * 100.).max(8.) / 100.) as f32;
            ui.painter().rect_filled(
                Rect::from_min_size(
                    egui::pos2(r.left() + i as f32 * (width + 4.), r.bottom() - height),
                    egui::vec2(width, height),
                ),
                0,
                ACCENT.gamma_multiply(0.8),
            );
        }
        mono(
            ui,
            format!(
                "свет мира {}° · оценка {}° · шум {}°{}{}",
                decimal(v.evaluator.light_true, 1),
                decimal(v.evaluator.light_estimated, 1),
                decimal(v.noise, 0),
                if v.swapped {
                    " · цвета тел обменены"
                } else {
                    ""
                },
                if e.overconfident {
                    " · штраф за самоуверенность"
                } else {
                    ""
                }
            ),
        );
    });
}
pub(crate) fn episode(ui: &mut Ui, s: &State, seed_input: &mut String, actions: &mut Vec<Action>) {
    card(ui, |ui| {
        heading(ui, "Эпизод");
        if let Some((_, _, hint)) = labels::SCENARIOS
            .iter()
            .find(|(id, _, _)| *id == s.scenario)
        {
            muted(ui, *hint);
        }
        ui.horizontal_wrapped(|ui| {
            for (id, label, _) in labels::SCENARIOS {
                if pill(ui, label, id == s.scenario, true) {
                    actions.push(Action::Scenario(id));
                }
            }
        });
        ui.horizontal_wrapped(|ui| {
            let label = ui.label(RichText::new("Зерно генератора").color(MUTED));
            let input = ui
                .add(
                    egui::TextEdit::singleline(seed_input)
                        .desired_width(75.)
                        .font(FontId::monospace(14.)),
                )
                .labelled_by(label.id);
            if input.gained_focus() {
                let mut state =
                    egui::text_edit::TextEditState::load(ui.ctx(), input.id).unwrap_or_default();
                state
                    .cursor
                    .set_char_range(Some(egui::text::CCursorRange::two(
                        egui::text::CCursor::new(0),
                        egui::text::CCursor::new(seed_input.chars().count()),
                    )));
                state.store(ui.ctx(), input.id);
            }
            if input.changed() {
                actions.push(Action::Seed(seed_input.clone()));
            }
        });
        let v = s.view();
        let e = &v.evaluator;
        egui::Frame::new()
            .fill(BG)
            .stroke(Stroke::new(1., LINE))
            .corner_radius(8)
            .inner_margin(12)
            .show(ui, |ui| {
                small(ui, "Только оценщик".to_uppercase());
                let retained = match e.retained {
                    None => "ещё рано",
                    Some(true) => "удержана",
                    Some(false) => "потеряна",
                };
                ui.label(format!(
                    "Идентичность A: {retained}{}",
                    e.initial_track_id
                        .map_or(String::new(), |id| format!(" · стартовая гипотеза T{id}"))
                ));
                for o in &e.objects {
                    mono(
                        ui,
                        format!(
                            "{} {} · гипотеза {}{}",
                            o.id,
                            if o.vanished {
                                "исчез"
                            } else if o.occluded {
                                "скрыт"
                            } else {
                                "виден"
                            },
                            o.nearest_track.map_or("нет".into(), |id| format!("T{id}")),
                            o.track_hue.map_or(String::new(), |h| format!(
                                " · оттенок {}",
                                jscompat::number(jscompat::round(h))
                            ))
                        ),
                    );
                }
            });
    });
}
pub(crate) fn interventions(ui: &mut Ui, s: &State, actions: &mut Vec<Action>) {
    card(ui, |ui| {
        heading(ui, "Вмешательства");
        muted(
            ui,
            "Режимы можно переключать во время эпизода. Изменение вступает в силу на следующем шаге.",
        );
        ui.horizontal_wrapped(|ui| {
            for (id, label) in labels::POLICIES {
                if pill(ui, label, s.policy == id, true) {
                    actions.push(Action::Policy(id));
                }
            }
        });
        ui.horizontal_wrapped(|ui| {
            if pill(ui, "Фиксация взгляда", s.pin_gaze, true) {
                actions.push(Action::Pin);
            }
            if pill(
                ui,
                "Подменить уверенность",
                s.inject && s.schema_enabled(),
                s.schema_enabled(),
            ) {
                actions.push(Action::Inject);
            }
            if pill(ui, "Повтор записи", s.replay, true) {
                actions.push(Action::Replay);
            }
        });
        let label = ui.label(
            RichText::new(format!("Дополнительный шум оттенка: {}°", s.extra_noise)).color(MUTED),
        );
        let mut noise = s.extra_noise;
        ui.spacing_mut().slider_width = ui.available_width();
        if ui
            .add(
                egui::Slider::new(&mut noise, 0..=48)
                    .show_value(false)
                    .step_by(1.),
            )
            .labelled_by(label.id)
            .changed()
        {
            actions.push(Action::Noise(noise));
        }
    });
}

pub(crate) fn protocol(ui: &mut Ui, s: &State) {
    card(ui, |ui| {
        heading(ui, "Протокол");
        let report = if let crate::BenchState::Ready(r) = &s.bench {
            Some(r)
        } else {
            None
        };
        let pending = s.bench.pending_label();
        muted(
            ui,
            if report.is_some() {
                "Доли успеха приведены с 95%-ми интервалами Уилсона. Доля шагов обхода рассчитана по всем шагам обеих задач вместе."
            } else {
                pending
            },
        );
        muted(ui, lab::criterion(200));
        if let Some(r) = report {
            for message in &r.saturation {
                muted(ui, message);
            }
            let note = lab::exploration_note(&r.rows);
            if !note.is_empty() {
                muted(ui, note);
            }
        }
        muted(ui, lab::AGE_ASSUMPTION);
        muted(
            ui,
            "Версия A без обучения: проверяет функцию дирижёра (CTC), но не гипотезу Genesis.",
        );
        egui::ScrollArea::horizontal()
            .id_salt("protocol-table")
            .show(ui, |ui| {
                ui.set_min_width(1042.);
                egui::Grid::new("results")
                    .spacing(egui::vec2(12., 12.))
                    .show(ui, |ui| {
                        let widths = [160., 170., 170., 175., 105., 95., 95.];
                        for (i, text) in [
                            "Условие",
                            "Удержание A",
                            "Пара различена",
                            "Разница с первым порядком, п. п. (A / пара)",
                            "Доля шагов обхода",
                            "Ошибка самомодели",
                            "Ошибка действия",
                        ]
                        .iter()
                        .enumerate()
                        {
                            ui.vertical(|ui| {
                                ui.set_width(widths[i]);
                                small(ui, text.to_uppercase());
                            });
                        }
                        ui.end_row();
                        if let Some(r) = report {
                            let first = r.rows.iter().find(|row| row.id == lab::Policy::FirstOrder);
                            for row in &r.rows {
                                ui.vertical(|ui| {
                                    ui.set_width(widths[0]);
                                    ui.label(row.label);
                                    small(ui, row.detail);
                                });
                                for text in [
                                    lab::format_rate(row.retention),
                                    lab::format_rate(row.pair),
                                    first.map_or("–".into(), |f| {
                                        format!(
                                            "{} / {}",
                                            lab::points(row.retention.rate - f.retention.rate),
                                            lab::points(row.pair.rate - f.pair.rate)
                                        )
                                    }),
                                    format!("{}%", decimal(row.explore_rate * 100., 2)),
                                    decimal(row.self_error, 3),
                                    decimal(row.action_error, 3),
                                ]
                                .into_iter()
                                .enumerate()
                                {
                                    ui.vertical(|ui| {
                                        ui.set_width(widths[text.0 + 1]);
                                        mono(ui, text.1);
                                    });
                                }
                                ui.end_row();
                            }
                        }
                    });
            });
        let result = |ui: &mut Ui, title: &str, text: &str| {
            egui::Frame::new()
                .stroke(Stroke::new(1., LINE))
                .corner_radius(8)
                .inner_margin(12)
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    small(ui, title);
                    ui.label(text);
                });
        };
        if ui.available_width() >= 590. {
            ui.columns(2, |cols| {
                result(
                    &mut cols[0],
                    "Причинная роль модели второго порядка",
                    report.map_or(pending, |r| lab::causal_label(r.causal)),
                );
                result(
                    &mut cols[1],
                    "Альтернативное объяснение правилом первого порядка",
                    report.map_or(pending, |r| {
                        if r.alternative_excluded {
                            "исключено"
                        } else {
                            "не исключено"
                        }
                    }),
                );
            });
        } else {
            result(
                ui,
                "Причинная роль модели второго порядка",
                report.map_or(pending, |r| lab::causal_label(r.causal)),
            );
            result(
                ui,
                "Альтернативное объяснение правилом первого порядка",
                report.map_or(pending, |r| {
                    if r.alternative_excluded {
                        "исключено"
                    } else {
                        "не исключено"
                    }
                }),
            );
        }
        egui::Frame::new().stroke(Stroke::new(1.,LINE)).corner_radius(8).inner_margin(12).show(ui,|ui| {
            ui.set_width(ui.available_width());small(ui,"Обнаружение повтора по собственному прогнозу");
            if let Some(r)=report {
                for replay in &r.replay {
                    ui.label(format!("{} · {} · AUC {}",replay.label,if replay.detected {"обнаружен"} else {"не обнаружен"},decimal(replay.auc,3)));
                    small(ui,format!("Средняя ошибка: {} в живом эпизоде, {} при повторе. Медиана парной разницы: {}. Пар с большей ошибкой при повторе: {}.",decimal(replay.live_error,3),decimal(replay.replay_error,3),decimal(replay.median_difference,3),lab::format_rate(replay.higher)));
                }
                if r.replay.iter().all(|r|r.detected){ui.label(lab::SHARED_DETECTION);}
            } else {ui.label(pending);}
        });
        ui.columns(2, |cols| {
            result(&mut cols[0], "Механизм", "реализован");
            result(&mut cols[1], "Феноменальное сознание", "не установлено");
        });
    });
}
