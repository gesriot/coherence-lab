use crate::state::State;
use lab::{Agreement, Look, Policy, Reason, Scenario, Snapshot, Source};
pub const SCENARIOS: [(Scenario, &str, &str); 5] = [
    (
        Scenario::Continuity,
        "Непрерывность",
        "Диск A уходит за стену. После его появления оценщик проверяет, сохранился ли номер трека.",
    ),
    (
        Scenario::Vanish,
        "Исчезновение",
        "Диск A исчезает на открытом месте. Оценщик проверяет, как долго сохраняется его трек.",
    ),
    (
        Scenario::Binding,
        "Связывание",
        "На шаге 16 цвета A и B меняются. Номер гипотезы должен остаться у тела, оттенок – обновиться.",
    ),
    (
        Scenario::Conflict,
        "Конфликт",
        "Два объекта похожих оттенков находятся в стороне от большого квадрата. Пара засчитывается, если ближайшие к A и B треки различаются шесть шагов подряд.",
    ),
    (
        Scenario::Nursery,
        "Свободная сцена",
        "Три тела и невысокая стена. Удобно менять шум и проверять, что происходит при фиксации взгляда или повторе записи.",
    ),
];
pub const POLICIES: [(Policy, &str); 6] = [
    (Policy::Full, "Полный контур"),
    (Policy::FirstOrder, "Первый порядок"),
    (Policy::Novelty, "Новизна"),
    (Policy::RecordOnly, "Только запись"),
    (Policy::NoSchema, "Без модели"),
    (Policy::DummyMemory, "Лишняя память"),
];
pub fn agreement(value: Agreement) -> &'static str {
    match value {
        Agreement::Settled => "согласовано",
        Agreement::Competing => "гипотезы конкурируют",
        Agreement::Underdetermined => "недоопределено",
        _ => "–",
    }
}
pub fn source(value: Source) -> &'static str {
    match value {
        Source::Observed => "наблюдение",
        Source::Predicted => "прогноз",
        Source::Remembered => "воспоминание",
        Source::Mixed => "смесь",
        _ => "–",
    }
}
pub fn reason(value: Reason) -> &'static str {
    match value {
        Reason::None => "ясно",
        Reason::Noise => "шум",
        Reason::Occlusion => "окклюзия",
        Reason::Conflict => "конфликт",
        Reason::Absence => "пусто, хотя ждали",
        Reason::NoData => "вне окна",
        _ => "–",
    }
}
pub fn attention(value: Look) -> &'static str {
    match value {
        Look::Conflict => "развести конфликт",
        Look::Verify => "проверить прогноз",
        Look::Uncertainty => "снять неопределённость",
        Look::Explore => "обход",
        Look::Salience => "к самому крупному пятну",
        Look::Pursue => "сопровождение",
        Look::Pinned => "взгляд зафиксирован",
        Look::Idle => "ожидание",
        _ => "–",
    }
}
pub fn status(state: &State) -> &'static str {
    status_parts(
        state.view().coupling == "broken",
        state.pin_gaze,
        state.inject,
        state.policy,
    )
}
fn status_parts(broken: bool, pin: bool, inject: bool, policy: Policy) -> &'static str {
    if broken {
        "Связь действие–вход разорвана"
    } else if pin {
        "Взгляд зафиксирован"
    } else if inject && matches!(policy, Policy::Full | Policy::RecordOnly) {
        "Уверенность подменена"
    } else {
        match policy {
            Policy::Full => "Контур замкнут",
            Policy::RecordOnly => "Модель не влияет",
            Policy::DummyMemory => "Память без модели",
            _ => POLICIES
                .iter()
                .find(|(p, _)| *p == policy)
                .map_or("–", |(_, label)| label),
        }
    }
}
pub fn step(view: &Snapshot<'_>) -> String {
    format!(
        "шаг {} · итерации {}/{} · {}",
        view.step,
        view.iterations_used,
        view.iteration_budget,
        agreement(view.attention.agreement)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn status_priority_covers_every_mode_and_intervention_combination() {
        for (mode, label) in POLICIES {
            for broken in [false, true] {
                for pin in [false, true] {
                    for inject in [false, true] {
                        let expected = match (broken, pin, inject, mode) {
                            (true, _, _, _) => "Связь действие–вход разорвана",
                            (_, true, _, _) => "Взгляд зафиксирован",
                            (_, _, true, Policy::Full | Policy::RecordOnly) => {
                                "Уверенность подменена"
                            }
                            (_, _, _, Policy::Full) => "Контур замкнут",
                            (_, _, _, Policy::RecordOnly) => "Модель не влияет",
                            (_, _, _, Policy::DummyMemory) => "Память без модели",
                            _ => label,
                        };
                        assert_eq!(status_parts(broken, pin, inject, mode), expected);
                    }
                }
            }
        }
    }
    #[test]
    fn status_uses_selected_mode_while_step_label_uses_current_snapshot() {
        let mut s = State::default();
        let label = step(&s.view());
        s.act(
            crate::Action::Policy(Policy::FirstOrder),
            std::time::Duration::ZERO,
        );
        assert_eq!(status(&s), "Первый порядок");
        assert_eq!(step(&s.view()), label);
        assert_eq!(s.view().ablation, Policy::Full);
    }
}
