use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

pub(crate) fn grad_enabled() -> bool {
    GRAD_ENABLED.with(|enabled| enabled.get())
}

pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    GRAD_ENABLED.with(|enabled| {
        let previous = enabled.replace(false);

        let result = f();

        enabled.set(previous);

        result
    })
}
