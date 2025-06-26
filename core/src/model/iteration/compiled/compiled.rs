pub trait Compiled<P, I> {
    fn compile(&self, params: &P) -> I;
}