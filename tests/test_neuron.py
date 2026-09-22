"""hypernix.neuron — training small networks that act.

What these tests are actually for
---------------------------------
An RL or imitation implementation is unusually easy to write wrong and
have it *look* right: the loop runs, the loss falls, and the policy is
useless. Mocked environments make that worse, because a mock returns
whatever the test author expected, so a sign error in a policy gradient
passes.

So nothing here is mocked. Every trainer is run against a real
environment with a known optimum, and the assertion is on the score it
reaches — "within a whisker of the expert", which is checkable — rather
than on "loss went down", which is not.

Budgets are small on purpose. These run in the ordinary suite, so each
trainer gets the smallest number of steps that still reliably separates
"learned it" from "did not".
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hypernix.neuron import nets, supervised  # noqa: E402
from hypernix.neuron.data import (  # noqa: E402
    Demonstrations,
    ReplayBuffer,
    Transition,
    discounted_returns,
    record_expert,
)
from hypernix.neuron.envs import (  # noqa: E402
    BalanceBeam,
    GridWorld,
    Space,
    rollout,
)
from hypernix.neuron.evaluate import compare, evaluate_policy  # noqa: E402
from hypernix.neuron.imitation import clone, dagger  # noqa: E402
from hypernix.neuron.rl import dqn, reinforce  # noqa: E402

# ---------------------------------------------------------------------------
# Spaces and environments
# ---------------------------------------------------------------------------


class TestSpaces:
    def test_a_space_is_discrete_or_continuous_not_both(self):
        """Both set is a bug in the caller, and one that produces a
        network with the wrong input width rather than an error."""
        with pytest.raises(ValueError):
            Space(n=4, shape=(3,))

    def test_a_space_has_to_be_something(self):
        with pytest.raises(ValueError):
            Space()

    def test_size_is_what_a_first_layer_needs(self):
        assert Space(shape=(3, 4)).size == 12
        assert Space(n=7).size == 7


class TestGridWorld:
    def test_the_same_seed_gives_the_same_episode(self):
        """Every score in this package is a mean over seeded episodes.
        If a seed does not pin the episode, none of them mean anything."""
        first = GridWorld().reset(seed=7)[0]
        second = GridWorld().reset(seed=7)[0]
        assert np.array_equal(first, second)

    def test_it_never_starts_on_the_goal(self):
        """A zero-step episode would score a perfect return for doing
        nothing, and would do it silently."""
        env = GridWorld()
        for seed in range(50):
            env.reset(seed=seed)
            assert env.optimal_steps() > 0

    def test_the_expert_is_optimal(self):
        """The imitation tests are only meaningful if the thing being
        imitated is actually right."""
        env = GridWorld()
        for seed in range(20):
            env.reset(seed=seed)
            expected = env.optimal_steps()
            _obs, actions, _rewards, terminated = rollout(
                env, lambda _o: env.expert_action(), seed=seed
            )
            assert terminated
            assert len(actions) == expected

    def test_running_out_of_time_is_truncated_not_terminated(self):
        """Different things. Bootstrapping through a time limit is
        correct; bootstrapping through a terminal state is not."""
        env = GridWorld(size=5, max_steps=3)
        env.reset(seed=0)
        last = None
        for _ in range(3):
            last = env.step(0)          # walk into the top edge forever
        _obs, _reward, terminated, truncated, _ = last
        assert truncated and not terminated

    def test_walking_into_a_wall_costs_and_does_not_move(self):
        env = GridWorld(size=3, walls=((1, 1),))
        env.reset(seed=0)
        env._agent = (0, 1)
        env._goal = (2, 2)
        _obs, reward, _t, _tr, _ = env.step(1)      # down, into the wall
        assert env._agent == (0, 1)
        assert reward < env.step_cost

    def test_an_action_outside_the_four_is_refused(self):
        env = GridWorld()
        env.reset(seed=0)
        with pytest.raises(ValueError):
            env.step(9)


class TestBalanceBeam:
    def test_it_never_starts_perfectly_centred(self):
        """A dead start would let a do-nothing policy score perfectly."""
        env = BalanceBeam()
        for seed in range(30):
            env.reset(seed=seed)
            assert abs(env._position) > 1e-9

    def test_the_expert_survives_the_full_episode(self):
        env = BalanceBeam(max_steps=200)
        _obs, actions, _rewards, terminated = rollout(
            env, lambda _o: env.expert_action(), seed=3
        )
        assert not terminated          # never fell off
        assert len(actions) == 200

    def test_falling_off_terminates(self):
        env = BalanceBeam(max_steps=10_000)
        env.reset(seed=0)
        terminated = False
        for _ in range(2000):
            _o, _r, terminated, _tr, _ = env.step(1)   # push one way forever
            if terminated:
                break
        assert terminated


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class TestData:
    def test_returns_are_reward_to_go(self):
        """Not the episode total on every step: crediting a step with
        reward collected before it is pure variance."""
        assert np.allclose(discounted_returns([0.0, 0.0, 1.0], 0.9),
                           [0.81, 0.9, 1.0])

    def test_only_the_last_step_of_a_terminated_episode_is_terminal(self):
        from hypernix.neuron.data import Trajectory

        trajectory = Trajectory(
            observations=[np.zeros(2, dtype=np.float32) for _ in range(3)],
            actions=[0, 1, 0],
            rewards=[0.0, 0.0, 1.0],
            terminated=True,
        )
        flags = [t.terminated for t in trajectory.transitions()]
        assert flags == [False, False, True]

    def test_a_truncated_episode_has_no_terminal_step(self):
        """The whole point of keeping the two apart."""
        from hypernix.neuron.data import Trajectory

        trajectory = Trajectory(
            observations=[np.zeros(2, dtype=np.float32) for _ in range(3)],
            actions=[0, 1, 0],
            rewards=[0.0, 0.0, 0.0],
            terminated=False,
        )
        assert not any(t.terminated for t in trajectory.transitions())

    def test_splitting_shuffles_first(self):
        """Demonstrations arrive in episode order. Holding out the last
        20% unshuffled holds out whole episodes, which reads as
        generalisation when it is not."""
        demonstrations = Demonstrations()
        for index in range(100):
            demonstrations.add([float(index)], index % 4)
        train, _held = demonstrations.split(0.2, seed=0)
        firsts = [float(o[0]) for o in train.observations[:20]]
        assert firsts != sorted(firsts)

    def test_an_empty_set_says_what_to_do(self):
        with pytest.raises(ValueError, match="record"):
            Demonstrations().tensors()

    def test_demonstrations_round_trip(self, tmp_path):
        env = GridWorld()
        demonstrations, _ = record_expert(env, episodes=5, seed=0)
        path = demonstrations.save(tmp_path / "d.npz")
        back = Demonstrations.load(path)
        assert back.actions == demonstrations.actions
        assert np.allclose(np.stack(back.observations),
                           np.stack(demonstrations.observations))

    def test_the_buffer_drops_the_oldest(self):
        buffer = ReplayBuffer(capacity=3)
        for index in range(5):
            buffer.add(
                Transition(np.zeros(2, dtype=np.float32), index, 0.0,
                           np.zeros(2, dtype=np.float32), False)
            )
        assert len(buffer) == 3
        assert [t.action for t in buffer] == [2, 3, 4]

    def test_sampling_an_empty_buffer_says_so(self):
        with pytest.raises(ValueError, match="empty"):
            ReplayBuffer().sample(1)

    def test_a_batch_bigger_than_the_buffer_still_works(self):
        """Refusing until the buffer is full would mean the opening
        steps of every run train on nothing."""
        buffer = ReplayBuffer(capacity=10)
        buffer.add(Transition(np.zeros(2, dtype=np.float32), 0, 1.0,
                              np.zeros(2, dtype=np.float32), True))
        observations, *_ = buffer.batch(8)
        assert observations.shape == (8, 2)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class TestNets:
    def test_it_reads_the_shapes_off_the_environment(self):
        model, spec = nets.for_env(GridWorld())
        assert spec.inputs == 4 and spec.outputs == 4

    def test_an_image_observation_gets_a_cnn(self):
        class Camera:
            observation_space = Space(shape=(3, 32, 32))
            action_space = Space(n=5)

        _model, spec = nets.for_env(Camera())
        assert spec.kind == "cnn"

    def test_the_cnn_survives_a_change_of_resolution(self):
        """Global pooling rather than a flatten, so the same spec works
        when somebody swaps the camera -- which always happens late."""
        model = nets.small_cnn((3, 32, 32), 4)
        assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 4)

    def test_a_checkpoint_carries_its_own_architecture(self, tmp_path):
        """A state_dict alone needs the script that made it, and that
        script is usually a notebook cell that is already gone."""
        model, spec = nets.for_env(GridWorld())
        path = nets.save(model, spec, tmp_path / "p.pt")
        back, back_spec = nets.load(path)
        assert back_spec.inputs == spec.inputs
        probe = torch.zeros(1, 4)
        assert torch.allclose(model(probe), back(probe))

    def test_a_bare_state_dict_is_refused_with_the_reason(self, tmp_path):
        path = tmp_path / "bare.pt"
        torch.save(nets.mlp(4, 4).state_dict(), path)
        with pytest.raises(ValueError, match="no spec"):
            nets.load(path)

    def test_greedy_is_deterministic(self):
        model, _ = nets.for_env(GridWorld())
        policy = nets.greedy(model)
        observation = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        assert len({policy(observation) for _ in range(10)}) == 1

    def test_an_unknown_activation_names_the_real_ones(self):
        with pytest.raises(ValueError, match="relu"):
            nets.mlp(4, 4, activation="banana")

    def test_a_continuous_action_space_is_refused_clearly(self):
        class Arm:
            observation_space = Space(shape=(6,))
            action_space = Space(shape=(3,))

        with pytest.raises(ValueError, match="continuous"):
            nets.for_env(Arm())


# ---------------------------------------------------------------------------
# Supervised
# ---------------------------------------------------------------------------


class TestSupervised:
    def test_it_learns_a_separable_problem(self):
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(400, 4, generator=generator)
        y = (x[:, 0] > 0).long()
        model = nets.mlp(4, 2, hidden=(32,))
        result = supervised.fit(model, x, y, epochs=60, seed=0)
        assert supervised.accuracy(model, x, y) > 0.95
        assert result.epochs_run > 0

    def test_it_keeps_the_best_epoch_not_the_last(self):
        generator = torch.Generator().manual_seed(1)
        x = torch.randn(120, 4, generator=generator)
        y = (x[:, 1] > 0).long()
        model = nets.mlp(4, 2, hidden=(32,))
        result = supervised.fit(
            model, x[:80], y[:80], validation=(x[80:], y[80:]),
            epochs=80, seed=0,
        )
        assert result.best_epoch <= result.epochs_run

    def test_stopping_watches_the_loss_not_the_accuracy(self):
        """The bug this replaced: on a 25-example holdout, accuracy only
        moves in steps of 1/25 and sits flat for ten epochs at a time
        while the model is still learning. Patience counted those flat
        epochs and stopped runs at epoch 41 with the training loss still
        falling -- producing a 48%-accurate policy on a task the same
        code solves completely."""
        env = GridWorld(size=5, max_steps=50)
        demonstrations, _ = record_expert(env, episodes=40, seed=0)
        model, result = clone(
            GridWorld(size=5, max_steps=50), demonstrations,
            epochs=200, seed=0,
        )
        assert result.action_accuracy > 0.9
        assert result.success_rate > 0.9

    def test_a_small_holdout_is_reported_as_such(self):
        generator = torch.Generator().manual_seed(2)
        x = torch.randn(60, 4, generator=generator)
        y = (x[:, 0] > 0).long()
        model = nets.mlp(4, 2, hidden=(16,))
        result = supervised.fit(
            model, x[:50], y[:50], validation=(x[50:], y[50:]),
            epochs=5, seed=0,
        )
        assert result.validation_is_small
        assert "coarse" in result.summary()

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="match"):
            supervised.fit(nets.mlp(2, 2), torch.zeros(4, 2), torch.zeros(3).long())


# ---------------------------------------------------------------------------
# Imitation
# ---------------------------------------------------------------------------


class TestImitation:
    def test_cloning_an_optimal_expert_reaches_the_goal(self):
        env = GridWorld(size=5, max_steps=50)
        demonstrations, trajectories = record_expert(env, episodes=60, seed=0)
        expert_return = sum(t.total_reward for t in trajectories) / len(trajectories)

        _model, result = clone(
            GridWorld(size=5, max_steps=50), demonstrations,
            epochs=200, evaluate_episodes=20, seed=0,
        )

        assert result.success_rate == 1.0
        # Within a whisker of the expert, not merely better than random.
        assert result.mean_return > expert_return - 0.05

    def test_it_reports_running_the_policy_not_just_copying_it(self):
        """Action accuracy and return are different measurements and
        come apart constantly. Both are reported."""
        env = GridWorld(size=5, max_steps=50)
        demonstrations, _ = record_expert(env, episodes=30, seed=0)
        _model, result = clone(
            GridWorld(size=5, max_steps=50), demonstrations, epochs=60, seed=0
        )
        assert 0.0 <= result.action_accuracy <= 1.0
        assert result.mean_return != result.action_accuracy

    def test_too_few_demonstrations_says_what_to_do(self):
        few = Demonstrations()
        few.add([0.0, 0.0, 0.0, 0.0], 0)
        with pytest.raises(ValueError, match="record more"):
            clone(GridWorld(), few)

    @pytest.mark.slow
    def test_dagger_beats_the_cloning_it_started_from(self):
        """The point of DAgger: round one is plain cloning on a handful
        of episodes and fails; aggregating the states the *policy*
        reaches is what fixes it."""
        env = GridWorld(size=5, max_steps=50)
        _model, result = dagger(
            env, rounds=3, episodes_per_round=8, epochs=120, seed=0
        )
        assert len(result.rounds) == 3
        assert result.rounds[-1]["success_rate"] > result.rounds[0]["success_rate"]
        assert result.rounds[-1]["demonstrations"] > result.rounds[0]["demonstrations"]

    def test_dagger_needs_an_expert_it_can_ask(self):
        class Silent:
            observation_space = Space(shape=(2,))
            action_space = Space(n=2)

            def reset(self, *, seed=None):
                return np.zeros(2, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(2, dtype=np.float32), 0.0, True, False, {}

        with pytest.raises(ValueError, match="query"):
            dagger(Silent())


# ---------------------------------------------------------------------------
# Reinforcement learning
# ---------------------------------------------------------------------------


class TestReinforcementLearning:
    @pytest.mark.slow
    def test_dqn_solves_the_sparse_reward_grid(self):
        """Sparse reward and terminal states: the combination that
        breaks an implementation which bootstraps through termination."""
        env = GridWorld(size=5, max_steps=50)
        _model, result = dqn(
            env, steps=6_000, seed=0, evaluate_every=1_500,
            evaluate_episodes=10,
        )
        assert result.reached(0.8), result.summary()
        assert result.best_success_rate == 1.0

    @pytest.mark.slow
    def test_reinforce_learns_to_balance(self):
        env = BalanceBeam(max_steps=200)
        _model, result = reinforce(
            env, episodes=200, batch_episodes=10, seed=0,
            evaluate_every=50, evaluate_episodes=5,
        )
        # Random play falls off almost immediately, scoring well under 50.
        assert result.best_return > 120.0, result.summary()

    def test_improved_says_so_when_a_run_starts_at_the_ceiling(self):
        """`improved` compares against the run's own past, so a task
        solved before the first checkpoint reports False. That is not a
        failure, and it is why `reached` exists."""
        from hypernix.neuron.rl import TrainResult

        flat = TrainResult(best_return=1.0)
        flat.history = [{"mean_return": 1.0}, {"mean_return": 1.0}]
        assert not flat.improved
        assert flat.reached(1.0)

    def test_a_run_with_no_checkpoints_has_not_improved(self):
        from hypernix.neuron.rl import TrainResult

        assert not TrainResult().improved


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_it_records_its_seeds(self):
        """"It got 0.82" is not a result anybody can check."""
        env = GridWorld()
        report = evaluate_policy(env, lambda _o: 0, episodes=5, seed=11)
        assert report.seeds == [11, 12, 13, 14, 15]

    def test_spread_separates_reliable_from_lucky(self):
        """Two policies with the same mean need different fixes."""
        env = GridWorld(size=5, max_steps=50)
        expert = evaluate_policy(
            env, lambda _o: env.expert_action(), episodes=20, seed=0
        )
        assert expert.success_rate == 1.0
        assert expert.return_spread < 0.1

    def test_comparison_uses_identical_episodes(self):
        """Comparing across different episodes is the commonest way to
        conclude a change helped when it did not."""
        env = GridWorld(size=5, max_steps=50)
        reports = compare(
            env,
            {"expert": lambda _o: env.expert_action(), "stuck": lambda _o: 0},
            episodes=6,
            seed=4,
        )
        assert reports["expert"].seeds == reports["stuck"].seeds
        assert reports["expert"].mean_return > reports["stuck"].mean_return

    def test_zero_episodes_is_refused(self):
        with pytest.raises(ValueError):
            evaluate_policy(GridWorld(), lambda _o: 0, episodes=0)


class TestTheBootstrap:
    """`td_target` alone, because the integration test does not catch it.

    Reintroducing the terminal-bootstrap bug and running
    `test_dqn_solves_the_sparse_reward_grid` passes: on a four-step grid
    with a 6,000-step budget the value estimate has not had time to
    diverge far enough to change the argmax. The bug is real and it is
    the commonest one in hand-written DQN, so the arithmetic is asserted
    directly instead of being hoped for through a policy score.
    """

    @staticmethod
    def _tensors(reward, value, terminated):
        return (
            torch.tensor([reward]),
            torch.tensor([value]),
            torch.tensor([float(terminated)]),
        )

    def test_a_terminal_step_is_worth_its_reward_and_nothing_more(self):
        from hypernix.neuron.rl import td_target

        rewards, values, terminated = self._tensors(1.0, 99.0, True)
        assert float(td_target(rewards, values, terminated, 0.99)) == pytest.approx(1.0)

    def test_a_non_terminal_step_bootstraps(self):
        from hypernix.neuron.rl import td_target

        rewards, values, terminated = self._tensors(1.0, 10.0, False)
        assert float(td_target(rewards, values, terminated, 0.5)) == pytest.approx(6.0)

    def test_a_time_limit_still_bootstraps(self):
        """The distinction that costs you the value of every state the
        clock happened to stop on. `truncated` never reaches here: the
        buffer records `terminated` alone."""
        from hypernix.neuron.data import Trajectory

        trajectory = Trajectory(
            observations=[np.zeros(2, dtype=np.float32)] * 2,
            actions=[0, 1],
            rewards=[1.0, 1.0],
            terminated=False,          # ran out of time
        )
        assert not any(t.terminated for t in trajectory.transitions())

    def test_mismatched_shapes_are_refused(self):
        from hypernix.neuron.rl import td_target

        with pytest.raises(ValueError, match="matching shapes"):
            td_target(torch.zeros(3), torch.zeros(2), torch.zeros(3), 0.99)
