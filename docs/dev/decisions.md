# Design decisions

Choices made where the paper (arXiv 2512.17992) does not specify the
design. Each entry says what was chosen and why.

## Table Clean Sim

- **Options.** The seven controllers PickToyFromTable(robot, toy, table),
  PlaceToyInBox(robot, toy, box), PickWiperFromBox(robot, wiper, box),
  PlaceWiperInBox(robot, wiper, box), PushBoxOut(robot, box),
  PullBoxIn(robot, box) and WipeTable(robot, wiper, table, box).
  WipeTable takes the box because its precondition "the box is on the far
  side" is about the box, and an operator can only mention its parameters.
- **Pushing and pulling the box** work with the wiper in hand. With only
  these seven options, a robot that needed an empty hand to push the box
  could never hold the wiper while the box is on the far side, which
  wiping needs.
- **Goal: the toys in the box, the table clean and the wiper back in the
  box.** The paper's goal leaves out the wiper. Without it, the
  demonstrations from the standard start (two toys on the table, the wiper
  in the box, the box on the near side) never use PullBoxIn or
  PlaceWiperInBox, so no operator can be learned for them, while the
  unseen test starts (wiper in hand, box on the far side) need both.
  Putting the wiper back makes every demonstration use all seven options,
  and every test start solvable with the learned operators. This also
  gives the paper's constraint "storing the towel requires the box on the
  near side" a role in every task.
- **State.** Per-object features as listed in the paper: 2D poses plus
  in_box, on_table and held for toys, in_box and held for the wiper, side
  for the box (0 near, 1 far), is_clean for the table and gripper for the
  robot.
- **Given predicates** are the goal predicates (ToyInBox, TableClean,
  WiperInBox); HandEmpty, HoldingToy, HoldingWiper, ToyOnTable, BoxNear and
  BoxFar are invented, and NoToyOnTable is the derived predicate ForAll
  ?toy. Not ToyOnTable(?toy, ?table).
