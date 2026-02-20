# derived_registry.py
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple
from collections import defaultdict
import logging
from predicators.structs import NSRT, AbstractPolicy, DefaultState, \
    DummyOption, GroundAtom, Metrics, Object, OptionSpec, \
    ParameterizedOption, Predicate, State, STRIPSOperator, Task, Type, \
    _GroundNSRT, _GroundSTRIPSOperator, _Option

EvalFn = Callable[[Set[GroundAtom], Set[Object]], Set[GroundAtom]]

class DerivedRule:
    def __init__(self, head: Predicate, eval_fn: EvalFn, deps: Iterable[Predicate]):
        self.head = head
        self.eval_fn = eval_fn
        self.deps = tuple(deps)  


DERIVED_REGISTRY: Dict[Predicate, DerivedRule] = {}

from itertools import product

def get_objects_of_type(obj_type: Type, objects: Set[Object]) -> List[Object]:
    """Filter objects by type using is_instance check."""
    return [o for o in objects if o.is_instance(obj_type)]

def all_groundings_fn(pred: Predicate, objects: Set[Object]) -> Iterable[GroundAtom]:

    domains = [list(get_objects_of_type(t, objects)) for t in pred.types]
    # logging.info(f"Domains for {pred}: {domains}{objects}")
    if any(len(dom)==0 for dom in domains):
        return []  

    for combo in product(*domains):            
        yield GroundAtom(pred, combo)   
    
    # logging.info(f"Generated all groundings for {pred}")


def clear_registry():
    DERIVED_REGISTRY.clear()



def register_negation(head_neg: Predicate, base_pred: Predicate,
                      group_idx: Optional[int] = None):


    def _same_group_name(name_root: str, name: str, gidx: Optional[int]) -> bool:
       
        if gidx is None:
            return name.startswith(name_root)
        return name.startswith(f"{name_root}_{gidx}")

    def _atom_sig(a: GroundAtom) -> Tuple[Tuple[Object, ...]]:
       
        return (tuple(a.entities),)

    def _eval(base_atoms: Set[GroundAtom], objects: Set[Object]) -> Set[GroundAtom]:
        out: Set[GroundAtom] = set()
        base_root = getattr(base_pred, "name", str(base_pred))

       
        base_true_sigs: Set[Tuple[Tuple[Object, ...]]] = set()
        for a in base_atoms:
            aname = getattr(a.predicate, "name", str(a.predicate))
            if _same_group_name(base_root, aname, group_idx):
                base_true_sigs.add(_atom_sig(a))

       
        all_inst = list(all_groundings_fn(base_pred, objects))

        
        for ga in all_inst:
            if _atom_sig(ga) not in base_true_sigs:
                out.add(GroundAtom(head_neg, ga.entities))

        return out

    DERIVED_REGISTRY[head_neg] = DerivedRule(head_neg, _eval, [base_pred])


def register_quantified(head: Predicate, base_pred: Predicate,
                        quantified_var_idxs: List[int], quantifier: str,
                        group_idx: Optional[int] = None,
                        neg_after_quant: bool=False, inner_neg: bool=False):
    qset = set(quantified_var_idxs)

    def _same_group_name(name_root: str, name: str, gidx: Optional[int]) -> bool:
      
        if gidx is None:
            return name.startswith(name_root)
        return name.startswith(f"{name_root}_{gidx}")

    def _eval(base_atoms: Set[GroundAtom], objects: Set[Object]) -> Set[GroundAtom]:
        out: Set[GroundAtom] = set()
        base_root = getattr(base_pred, "name", str(base_pred))

       
        def atom_sig(a: GroundAtom) -> Tuple[Tuple[Object, ...]]:
            
            return (tuple(a.entities),)

        base_true_sigs: Set[Tuple[Tuple[Object, ...]]] = set()
        for a in base_atoms:
            aname = getattr(a.predicate, "name", str(a.predicate))
            if _same_group_name(base_root, aname, group_idx):
                base_true_sigs.add(atom_sig(a))

       
        all_inst = list(all_groundings_fn(base_pred, objects))

        
        if inner_neg:
            all_sigs = {atom_sig(ga) for ga in all_inst}
            base_true_sigs = all_sigs - base_true_sigs
            
        # logging.info(f"base_true: {base_true_sigs},all_sigs: {all_sigs}")
    
        bucket_all: Dict[Tuple[Object, ...], Set[GroundAtom]] = defaultdict(set)
        bucket_true: Dict[Tuple[Object, ...], Set[GroundAtom]] = defaultdict(set)

        for ga in all_inst:
            es = list(ga.entities)
            key_free = tuple(es[i] for i in range(len(es)) if i not in qset)
            bucket_all[key_free].add(ga)
            if atom_sig(ga) in base_true_sigs:
                bucket_true[key_free].add(ga)

       
        for key_free, all_set in bucket_all.items():
            true_set = bucket_true.get(key_free, set())
            if quantifier == "Exists":
                truth = (len(true_set) > 0)
            elif quantifier == "ForAll":
                truth = (len(all_set) > 0 and len(true_set) == len(all_set))
            else:
                raise ValueError(f"Unknown quantifier: {quantifier}")

            if neg_after_quant:
                truth = not truth

            if truth:
               
                out.add(GroundAtom(head, key_free if len(key_free) > 0 else ()))
        return out

    # logging.info(f"Registering quantified: head={head}, base={base_pred}, "
    #              f"quantifier={quantifier}, q_idxs={quantified_var_idxs}, group_idx={group_idx}, "
    #              f"neg_after_quant={neg_after_quant}, inner_neg={inner_neg}")
    DERIVED_REGISTRY[head] = DerivedRule(head, _eval, [base_pred])


def derive_closure(
    base_atoms: Set[GroundAtom],
    objects: Set[Object]):
    derived = set()
    specs = DERIVED_REGISTRY.values()
    if not specs:
        # logging.info("No derived predicates registered.")
        return set(base_atoms)
    for rule in specs:
        # logging.info(f"Applying rule for {rule.head.name} based on {[p.name for p in rule.deps]}")
        derived |= rule.eval_fn(base_atoms, objects)
    return set(base_atoms) | derived



