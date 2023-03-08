from typing import NamedTuple
import equinox as eqx
import optax
import dataclasses

class TrainState(eqx.Module):
    step: int
    params: eqx.Module
    tx: optax.GradientTransformation
    opt_state: optax.OptState
    
    @classmethod
    def create(cls, params: eqx.Module, tx: optax.GradientTransformation):
        return cls(step=0, params=params, opt_state=tx.init(params), tx=tx)
    
    def apply_gradients(self, grads):
        updates, opt_state = self.tx.update(grads, self.opt_state, params=self.params)
        params = eqx.apply_updates(self.params, updates)
        # return TrainState(step=self.step+1, params=params, tx=self.tx, opt_state=opt_state)
        return dataclasses.replace(self, step=self.step+1, params=params, opt_state=opt_state)
        