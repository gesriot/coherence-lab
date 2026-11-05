use crate::{Action, Speed, State, labels, panels, scene, theme::*};
use eframe::egui::{self, Rect, RichText, Ui};
use std::time::Duration;

pub struct Viewer {
    pub(crate) state: State,
    seed_input: String,
    pub(crate) scene_rect: Rect,
}
impl Viewer {
    pub fn new(ctx: &egui::Context) -> Self {
        let mut viewer = Self::waiting(ctx);
        viewer.state.start_bench();
        viewer
    }
    pub(crate) fn waiting(ctx: &egui::Context) -> Self {
        setup(ctx);
        Self {
            state: State::default(),
            seed_input: "4".into(),
            scene_rect: Rect::NOTHING,
        }
    }
    fn scene_card(&mut self, ui: &mut Ui, actions: &mut Vec<Action>) {
        egui::Frame::new().fill(SURFACE).stroke(egui::Stroke::new(1.,LINE)).corner_radius(16).show(ui,|ui| {
            egui::Frame::new().inner_margin(16).show(ui,|ui| {
                heading(ui,if self.state.show_truth {"Мир оценщика"} else {"Вход агента"});
                small(ui,if self.state.show_truth {"Подписи объектов доступны оценщику. Щелчок фиксирует взгляд."} else {"Цветные клетки показывают измерения в текущем окне. Щелчок фиксирует взгляд."});
                ui.horizontal_wrapped(|ui| {
                    if pill(ui,if self.state.running {"Пауза"} else {"Пуск"},true,true){actions.push(Action::ToggleRunning);}
                    if pill(ui,"Шаг",false,true){actions.push(Action::Step);}
                    if pill(ui,"Сначала",false,true){actions.push(Action::Reset);}
                    for (speed,label) in [(Speed::One,"1×"),(Speed::Two,"2×"),(Speed::Four,"4×")] {
                        if pill(ui,label,self.state.speed==speed,true){actions.push(Action::Speed(speed));}
                    }
                    if pill(ui,"Истина",self.state.show_truth,true){actions.push(Action::Truth);}
                    if pill(ui,"Гипотезы",self.state.show_hypotheses,true){actions.push(Action::Hypotheses);}
                });
            });
            ui.separator();
            let (rect,pick)=scene::draw(ui,&self.state.view(),self.state.show_truth,self.state.show_hypotheses);
            self.scene_rect=rect;
            if let Some([x,y])=pick {actions.push(Action::Pick(x,y));}
            egui::Frame::new().inner_margin(12).show(ui,|ui| {
                ui.horizontal_wrapped(|ui| {for label in ["квадрат – сенсорное окно","пунктир – гипотеза","кольцо – следующий взгляд","штриховка – стена"] {small(ui,label);}});
            });
        });
    }
    fn laboratory(&mut self, ui: &mut Ui, width: f32, wide: bool, actions: &mut Vec<Action>) {
        ui.set_width(width);
        ui.add_space(20.);
        let header = |ui: &mut Ui| {
            ui.label(
                RichText::new("Лаборатория, не личность".to_uppercase())
                    .monospace()
                    .size(12.)
                    .color(ACCENT),
            );
            ui.label(
                RichText::new("Когерентность").font(display(if width >= 600. { 48. } else { 36. })),
            );
            muted(
                ui,
                "Минимальный контур в духе Баха: сцена собирается из конкурирующих гипотез, а модель собственного восприятия выбирает, куда смотреть. Это не доказательство квалиа.",
            );
        };
        let status = |ui: &mut Ui| {
            egui::Frame::new()
                .fill(SURFACE)
                .stroke(egui::Stroke::new(1., LINE))
                .corner_radius(24)
                .inner_margin(10)
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(labels::status(&self.state))
                            .monospace()
                            .size(12.),
                    );
                });
            mono(ui, labels::step(&self.state.view()));
        };
        if width >= 750. {
            ui.columns(2, |cols| {
                // Columns are justified, and a justified label stretches its wrapped lines.
                cols[0].with_layout(egui::Layout::top_down(egui::Align::LEFT), header);
                cols[1].with_layout(egui::Layout::top_down(egui::Align::RIGHT), status);
            });
        } else {
            header(ui);
            status(ui);
        }
        ui.add_space(12.);
        ui.separator();
        ui.add_space(12.);
        let pipeline = [
            "Измерения",
            "Гипотезы",
            "Сцена",
            "Модель внимания",
            "Взгляд",
        ];
        let count = if width >= 590. { 5 } else { 2 };
        for row in pipeline.chunks(count) {
            ui.columns(count, |cols| {
                for (i, label) in row.iter().enumerate() {
                    let number = pipeline.iter().position(|x| x == label).unwrap_or(0) + 1;
                    egui::Frame::new()
                        .fill(SURFACE)
                        .stroke(egui::Stroke::new(1., LINE))
                        .corner_radius(12)
                        .inner_margin(10)
                        .show(&mut cols[i], |ui| {
                            ui.set_width(ui.available_width());
                            ui.horizontal_wrapped(|ui| {
                                ui.label(
                                    RichText::new(number.to_string()).monospace().color(ACCENT),
                                );
                                ui.label(*label);
                            });
                        });
                }
            });
        }
        ui.add_space(8.);
        muted(
            ui,
            "Воспринимающий – не отдельная функция и не это окно. Это вся петля: измерения уточняют гипотезы, гипотезы входят в ограниченную сцену, сцена обновляет модель внимания, модель двигает взгляд, взгляд меняет следующие измерения.",
        );
        ui.add_space(8.);
        if wide {
            ui.horizontal_top(|ui| {
                ui.allocate_ui_with_layout(
                    egui::vec2((width - 16.) * 0.6, 0.),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| self.scene_card(ui, actions),
                );
                ui.add_space(8.);
                ui.allocate_ui_with_layout(
                    egui::vec2((width - 16.) * 0.4, 0.),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        panels::present(ui, &self.state.view());
                        ui.add_space(8.);
                        panels::attention(ui, &self.state.view());
                    },
                );
            });
        } else {
            self.scene_card(ui, actions);
            panels::present(ui, &self.state.view());
            panels::attention(ui, &self.state.view());
        }
        ui.add_space(8.);
        panels::coherence(ui, &self.state.view());
        ui.add_space(8.);
        if wide {
            ui.columns(2, |cols| {
                panels::episode(&mut cols[0], &self.state, &mut self.seed_input, actions);
                panels::interventions(&mut cols[1], &self.state, actions);
            });
        } else {
            panels::episode(ui, &self.state, &mut self.seed_input, actions);
            panels::interventions(ui, &self.state, actions);
        }
        ui.add_space(8.);
        panels::protocol(ui, &self.state);
        ui.add_space(80.);
    }
    pub(crate) fn content(&mut self, ui: &mut Ui) {
        let now = Duration::from_secs_f64(ui.input(|i| i.time).max(0.));
        self.state.tick(now);
        if let Some(delay) = self.state.repaint_after(now) {
            ui.ctx().request_repaint_after(delay);
        }
        if self.state.worker_pending() {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
        let wide = ui.ctx().content_rect().width() >= 1024.;
        let mut actions = Vec::new();
        egui::Frame::new().fill(BG).show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("laboratory")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let margin = if ui.ctx().content_rect().width() >= 640. {
                        24.
                    } else {
                        16.
                    };
                    let width = (ui.available_width() - 2. * margin).clamp(0., 1104.);
                    let left = ((ui.available_width() - width) / 2.).max(0.);
                    ui.horizontal_top(|ui| {
                        ui.add_space(left);
                        ui.vertical(|ui| {
                            self.laboratory(ui, width, wide, &mut actions);
                        });
                    });
                });
        });
        for action in actions {
            self.state.act(action, now);
            self.seed_input = self.state.seed.to_string();
            ui.ctx().request_repaint();
        }
    }
}
impl eframe::App for Viewer {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        self.content(ui);
    }
}

#[cfg(test)]
mod tests;
