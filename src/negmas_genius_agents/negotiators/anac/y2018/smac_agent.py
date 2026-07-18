"""SMACAgent from ANAC 2018.

This module implements SMACAgent, a Genius agent whose parameters were
tuned offline (using the SMAC hyper-parameter optimizer) for different
combinations of domain size, reservation value and discount factor. The
agent then selects one of 15 pre-computed parameter sets at negotiation
start based on those domain characteristics.

References:
    - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
    - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
      Competition (ANAC 2018)." IJCAI 2019.
    - Genius framework: https://ii.tudelft.nl/genius/
    - Original package: agents.anac.y2018.smac_agent.SMAC_Agent
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from negmas.outcomes import Outcome
from negmas.preferences import BaseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from negmas_genius_agents.utils.outcome_space import SortedOutcomeSpace

if TYPE_CHECKING:
    from negmas.situated import Agent
    from negmas.negotiators import Controller

__all__ = ["SMACAgent"]


# ---------------------------------------------------------------------------
# Pre-tuned parameter tables (copied verbatim from the original Java agent).
# The parameter set is selected using ``parameterIndex = 5 * domainIndex +
# resdisIndex`` where ``domainIndex`` depends on the domain size and
# ``resdisIndex`` depends on the reservation value / discount factor.
# ---------------------------------------------------------------------------

BS_UTIL_INIT = [
    0.8726814917149487, 0.777828754759162, 0.794611415535526, 0.8338477204370194,
    0.794611415535526, 0.9902714777558712, 0.8177264028188669, 0.9369503033177021,
    0.8752031270267119, 0.9130860700146172, 0.8985462879619349, 0.968496462717577,
    0.9475036977013452, 0.6358476408988469, 0.8187434422624748,
]
BS_UTIL_END = [
    0.7234378711390601, 0.5039748507682985, 0.28288734275992156, 0.5043866091578793,
    0.28288734275992156, 0.4003485368855013, 0.5775989724689408, 0.8449151525540153,
    0.6439901974110889, 0.5128376365334515, 0.8815381590699309, 0.7512965956758736,
    0.7732741878711409, 0.6890153697768392, 0.4575950773478153,
]
BS_SHAPE_LEFT = [
    -3.3591096890191987, -1.701604675744841, -1.477010357010462, -3.8655124599356254,
    -1.477010357010462, -4.578937298449693, -0.6220720689296879, -0.33672176040647983,
    -1.028771048186937, -2.0559509070865096, -2.8918016435189267, -0.7799716189241472,
    -1.3695282977990084, -5.218031753259722, -2.751303202136022,
]
BS_SHAPE_RIGHT = [
    1.6257316909626451, 5.743207917861144, 0.5042105201178133, 1.2177749917981506,
    0.5042105201178133, 0.23509522311374287, 4.091234186082431, 4.1350076547071914,
    2.4754885878922317, 2.164288746994306, 2.7254163735351717, 4.6865277838004795,
    3.2756425247103667, 4.954211560087625, 0.9268696757205295,
]
BS_STD_DEV = [
    0.011291090432680746, 0.00806602340055852, 0.03993171716605801, 0.04541686791307292,
    0.03993171716605801, 0.016756394004539998, 0.02050154318077101, 0.029271019497824725,
    0.043415206320189886, 0.0070856192273503205, 0.039755330055406116, 0.016399241309376957,
    0.025102281569543264, 0.04820460835411705, 0.02050857459278594,
]
BS_UTIL_RANGE = [
    0.12747022094357666, 0, 0, 0, 0, 0.13272162785215005, 0, 0.0983190266452012,
    0.05856974543348425, 0, 0.07746667826547407, 0.07280579445258521,
    0.18730438418792866, 0.14765559656825974, 0.1,
]
BS_USE_UTIL_RANGE = [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1]
BS_SCALE_MIN_MAX = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

AS_TIME_THRESHOLD1 = [
    0.04674955360732539, 0.21892573208596458, 0.16179316499336657, 0.017950789417510913,
    0.16179316499336657, 0.24113343505148666, 0.07195359710638492, 0.24062799038166227,
    0.06459440421448916, 0.05992314378385267, 0.0404833611227041, 0.016365548306668937,
    0.03720304384144093, 0.03478252915311665, 0.09717022719016756,
]
AS_TIME_THRESHOLD2 = [
    0.43303206747881884, 0.43815743675678354, 0.661647912300392, 0.6641557212474182,
    0.661647912300392, 0.6343062109766446, 0.31691596321099397, 0.6193435123380937,
    0.5101124045982374, 0.6439117927197825, 0.3097759359131537, 0.6779137891044901,
    0.3191532652234305, 0.36151362475862164, 0.49570228758309676,
]
AS_TIME_THRESHOLD3 = [
    0.7245716275829817, 0.9673408842892797, 0.9430960906851461, 0.8068138909742011,
    0.9430960906851461, 0.9259546534020413, 0.7785900567996049, 0.7315861505357254,
    0.9861492577603947, 0.8404594822840576, 0.8996360620044062, 0.8866203037044917,
    0.7572372266466516, 0.9713328720627714, 0.7682567113937757,
]
AS_RATE1 = [
    0.11219220522147239, 0.08003925149326611, 0.33051281601825294, 0.5168640464469102,
    0.33051281601825294, 0.13512460983555102, 0.3688780153278271, 0.29676619243083197,
    0.6646699560793469, 0.08650858607226897, 0.5000849837001996, 0.6245922813980603,
    0.40987041296162613, 0.5267683830657954, 0.42081900454556703,
]
AS_RATE2 = [
    0.004898877513807327, 0.5634389184435432, 0.5882550637609057, 0.1736660799531687,
    0.5882550637609057, 0.3052688029644835, 0.2634165350102941, 0.1128289859613561,
    0.09148496643943949, 0.594472202286485, 0.003764210630213882, 0.3267176154876564,
    0.16025942852106756, 0.33889339364499194, 0.29231207452138,
]
AS_RATE3 = [
    0.548875357115968, 0.4749109394742966, 0.14967914649484768, 0.12365288690420985,
    0.14967914649484768, 0.46068703380168724, 0.008086833565920028, 0.4505848794497385,
    0.5688130910774031, 0.009397472597385702, 0.3072190282216138, 0.013929751760161762,
    0.22948249946166988, 0.24981417826220348, 0.02859766493418777,
]
AS_DISCOUNT_RATE1 = [
    0.60078421520277, 0.041443536726597396, 0.062408258801741855, 0.09097496706566402,
    0.062408258801741855, 0.2201074114341298, 0.6400322557728236, 0.5788620610232695,
    0.11081606927344186, 0.5229841600644036, 0.24433177863341993, 0.4661668394745566,
    0.35628521577122196, 0.33720834099081726, 0.049371938906709045,
]
AS_DISCOUNT_RATE2 = [
    0.60078421520277, 0.041443536726597396, 0.062408258801741855, 0.09097496706566402,
    0.062408258801741855, 0.2201074114341298, 0.6400322557728236, 0.5788620610232695,
    0.11081606927344186, 0.5229841600644036, 0.3208622393478284, 0.4661668394745566,
    0.35628521577122196, 0.33720834099081726, 0.049371938906709045,
]
AS_DISCOUNT_RATE3 = [
    0.7657680975742767, 0.39061103724559, 0.2228029756845122, 0.23097016208565402,
    0.2228029756845122, 0.6512371835270208, 0.6855684205205383, 0.051066815285715085,
    0.3292575463583758, 0.5720037378648264, 0.40017902857747045, 0.2710384577710453,
    0.25373107722701543, 0.36883611624486773, 0.10017027025202459,
]
AS_THRESHOLD1 = [
    0.8219843158180532, 0.923099025516942, 0.7932091636158838, 0.9580958735836536,
    0.7932091636158838, 0.9685239192034572, 0.7859975846053607, 0.7499410639013572,
    0.7020301444423871, 0.7739229197734587, 0.9240012133461064, 0.7794051769942631,
    0.8620257399086155, 0.7607469411657954, 0.8870099642076714,
]

OMS_UTIL_INIT = [0, 0, 0, 0, 0, 0, 0, 0.815223752226039, 0, 0, 0, 0, 0, 0, 0.5163591000775694]
OMS_UTIL_END = [0, 0, 0, 0, 0, 0, 0, 0.6473775354471122, 0, 0, 0, 0, 0, 0, 0.7778975091828539]
OMS_SHAPE_LEFT = [0, 0, 0, 0, 0, 0, 0, -1.7202463296555042, 0, 0, 0, 0, 0, 0, -0.5137389957405025]
OMS_SHAPE_RIGHT = [0, 0, 0, 0, 0, 0, 0, 2.317670771610482, 0, 0, 0, 0, 0, 0, 5.911300168932384]
OMS_USE_SCALING = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
OMS_USE_OM = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
OMS_LIMIT_SCALING = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
OMS_EUCLIDEAN = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

USE_RES_VALUE = [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1]
RM_UPDATE_UTIL_CURVE = [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]

DOMAIN_SMALL_THRESHOLD = 100
DOMAIN_MEDIUM_THRESHOLD = 1000
RES_THRESHOLD = 0.5
DISCOUNT_THRESHOLD = 0.5


class _UtilCurve:
    """Sigmoid-shaped utility concession curve (port of Java's ``UtilCurve``)."""

    def __init__(
        self, shape_left: float, shape_right: float, util_init: float, util_end: float
    ):
        self.shape_left = shape_left
        self.shape_right = shape_right
        self.util_init = util_init
        self.util_end = util_end
        self.util_min = 0.0
        self.util_max = 1.0
        self.std_dev = 0.0

        self._correction = 1.0 / (1.0 + math.exp(-shape_left))
        denom = 1.0 / (1.0 + math.exp(-shape_right)) - self._correction
        self._correction_factor = (util_end - util_init) / denom if denom != 0 else 0.0

    def set_min_max(self, util_min: float, util_max: float) -> None:
        self.util_min = util_min
        self.util_max = util_max

    def set_std_dev(self, std_dev: float) -> None:
        self.std_dev = std_dev

    def set_min_accepted(self, min_accepted: float) -> None:
        final_util = self.util_min + (self.util_max - self.util_min) * self.util_end
        if min_accepted > final_util:
            denom = self.util_max - self.util_min
            self.util_end = (min_accepted - self.util_min) / denom if denom else self.util_end
        denom = 1.0 / (1.0 + math.exp(-self.shape_right)) - self._correction
        self._correction_factor = (
            (self.util_end - self.util_init) / denom if denom != 0 else 0.0
        )

    def get_util(self, time: float) -> float:
        x = (self.shape_right - self.shape_left) * time + self.shape_left
        s = (1.0 / (1.0 + math.exp(-x)) - self._correction) * self._correction_factor
        s += self.util_init
        noise = random.gauss(0.0, self.std_dev) if self.std_dev else 0.0
        return self.util_min + (self.util_max - self.util_min) * s + noise


class _AgentProfile:
    """Frequency-based opponent model (approximate port of ``AgentProfile``)."""

    def __init__(self) -> None:
        self._value_frequency: dict[int, dict] = {}
        self._issue_frequency: dict[int, int] = {}
        self._last_bid: Outcome | None = None
        self.best_accepted: Outcome | None = None
        self._bids_received = 0
        self._issues_changed = 0
        self._num_issues = 0

    def register_offer(self, bid: Outcome) -> None:
        self._bids_received += 1
        if self._num_issues == 0:
            self._num_issues = len(bid)
        if self._last_bid is None:
            self._last_bid = bid

        for i, value in enumerate(bid):
            self._value_frequency.setdefault(i, {})
            self._value_frequency[i][value] = self._value_frequency[i].get(value, 0) + 1
            if value != self._last_bid[i]:
                self._issues_changed += 1
                self._issue_frequency[i] = self._issue_frequency.get(i, 0) + 1
        self._last_bid = bid

    def register_accept(self, bid: Outcome, utility: float) -> None:
        if self.best_accepted is None:
            self.best_accepted = bid
        # Note: comparison uses caller-provided utility (self util space).

    def evaluate_bid(self, bid: Outcome) -> float:
        if self._bids_received == 0 or self._num_issues == 0:
            return 0.0
        predict = 0.0
        for i, value in enumerate(bid):
            f_value = self._value_frequency.get(i, {}).get(value, 0)
            f_issue = self._issue_frequency.get(i, 0)
            if self._issues_changed == 0:
                predict += (f_value / self._bids_received) * (1.0 / self._num_issues)
            else:
                predict += (
                    (f_value / self._bids_received)
                    * (1.0 - f_issue / self._issues_changed)
                    / (self._num_issues - 1.0)
                    if self._num_issues > 1
                    else 0.0
                )
        return predict

    @property
    def last_bid_opponent_utility(self) -> float:
        if self._last_bid is None:
            return 0.0
        return self.evaluate_bid(self._last_bid)


class SMACAgent(SAONegotiator):
    """
    SMACAgent from ANAC 2018.

    Note:
        This is an AI-generated reimplementation based on the original Java code
        from the Genius framework. It may not behave identically to the original.

    SMACAgent selects one of 15 offline-tuned parameter configurations
    (optimized with the SMAC hyper-parameter tuner) based on the domain's
    reservation value, discount factor and outcome-space size. The chosen
    parameters drive a sigmoid-shaped bidding-utility curve, a matching
    sigmoid-shaped acceptance strategy, and (for some configurations) a
    frequency-based opponent model used to select among candidate bids.

    References:
        - ANAC 2018: https://ii.tudelft.nl/negotiation/node/12
        - Baarslag, T., et al. (2019). "The Ninth Automated Negotiating Agents
          Competition (ANAC 2018)." IJCAI 2019.
        - Genius framework: https://ii.tudelft.nl/genius/
        - Original package: agents.anac.y2018.smac_agent.SMAC_Agent

    **Offering Strategy:**
        A sigmoid "bidding-strategy curve" (``BS``) maps relative time to a
        target utility. Bids are then selected from a widening utility range
        around that target using the opponent-model selection routine
        (``OMS``), falling back to a random choice among candidates when no
        opponent model is used for the selected configuration.

    **Acceptance Strategy:**
        A separate sigmoid-based threshold on the opponent's last offer with
        three time phases (each phase's threshold decays from the previous
        one using tuned rate/discount parameters), plus an unconditional
        acceptance if the opponent's offer exceeds a tuned high threshold.

    **Opponent Modeling:**
        Frequency-based per-issue-value model (only active for some tuned
        configurations); used within ``OMS`` to prefer bids whose predicted
        opponent utility is close to (but not above) a target opponent
        utility derived from the ``OMS`` sigmoid curve.

    Args:
        preferences: NegMAS preferences/utility function.
        ufun: Utility function (overrides preferences if given).
        name: Negotiator name.
        parent: Parent controller.
        owner: Agent that owns this negotiator.
        id: Unique identifier.
        **kwargs: Additional arguments passed to parent.
    """

    def __init__(
        self,
        preferences: BaseUtilityFunction | None = None,
        ufun: BaseUtilityFunction | None = None,
        name: str | None = None,
        parent: Controller | None = None,
        owner: Agent | None = None,
        id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            preferences=preferences,
            ufun=ufun,
            name=name,
            parent=parent,
            owner=owner,
            id=id,
            **kwargs,
        )
        self._outcome_space: SortedOutcomeSpace | None = None
        self._initialized = False

        self._res_value = 0.0
        self._discount_factor = 1.0

        self._bs_curve: _UtilCurve | None = None
        self._oms_curve: _UtilCurve | None = None

        # Selected (tuned) parameters
        self._bs_use_util_range = 0
        self._bs_util_range = 0.0
        self._as_tt1 = self._as_tt2 = self._as_tt3 = 0.0
        self._as_rate1 = self._as_rate2 = self._as_rate3 = 0.0
        self._as_dr1 = self._as_dr2 = self._as_dr3 = 0.0
        self._as_threshold1 = 0.0
        self._oms_use_om = 0

        self._agent_profile = _AgentProfile()

        self._my_last_bid: Outcome | None = None
        self._last_received_bid: Outcome | None = None
        self._max_bid: Outcome | None = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        if self._initialized:
            return
        if self.ufun is None:
            return

        self._outcome_space = SortedOutcomeSpace(ufun=self.ufun)
        outcomes = self._outcome_space.outcomes
        util_min = self._outcome_space.min_utility
        util_max = self._outcome_space.max_utility
        if outcomes:
            self._max_bid = outcomes[0].bid

        self._res_value = getattr(self.ufun, "reserved_value", None) or 0.0
        self._discount_factor = getattr(self.ufun, "discount_factor", 1.0) or 1.0
        domain_size = len(outcomes)

        idx = self._select_parameter_index(self._res_value, self._discount_factor, domain_size)

        if self._res_value > util_min and USE_RES_VALUE[idx] == 1:
            util_min = self._res_value

        self._bs_curve = _UtilCurve(
            BS_SHAPE_LEFT[idx], BS_SHAPE_RIGHT[idx], BS_UTIL_INIT[idx], BS_UTIL_END[idx]
        )
        self._oms_curve = _UtilCurve(
            OMS_SHAPE_LEFT[idx], OMS_SHAPE_RIGHT[idx], OMS_UTIL_INIT[idx], OMS_UTIL_END[idx]
        )

        if BS_SCALE_MIN_MAX[idx] == 1:
            self._bs_curve.set_min_max(util_min, util_max)
        self._bs_curve.set_std_dev(BS_STD_DEV[idx])

        self._bs_use_util_range = BS_USE_UTIL_RANGE[idx]
        self._bs_util_range = BS_UTIL_RANGE[idx]

        self._as_tt1 = AS_TIME_THRESHOLD1[idx]
        self._as_tt2 = AS_TIME_THRESHOLD2[idx]
        self._as_tt3 = AS_TIME_THRESHOLD3[idx]
        self._as_rate1 = AS_RATE1[idx]
        self._as_rate2 = AS_RATE2[idx]
        self._as_rate3 = AS_RATE3[idx]
        self._as_dr1 = AS_DISCOUNT_RATE1[idx]
        self._as_dr2 = AS_DISCOUNT_RATE2[idx]
        self._as_dr3 = AS_DISCOUNT_RATE3[idx]
        self._as_threshold1 = AS_THRESHOLD1[idx]

        self._oms_use_om = OMS_USE_OM[idx]

        self._initialized = True

    @staticmethod
    def _select_parameter_index(
        res_value: float, discount_factor: float, domain_size: float
    ) -> int:
        if res_value == 0 or discount_factor == 1:
            resdis_index = 0
        elif res_value <= RES_THRESHOLD and discount_factor <= DISCOUNT_THRESHOLD:
            resdis_index = 1
        elif res_value <= RES_THRESHOLD and discount_factor > DISCOUNT_THRESHOLD:
            resdis_index = 2
        elif res_value > RES_THRESHOLD and discount_factor <= DISCOUNT_THRESHOLD:
            resdis_index = 3
        else:
            resdis_index = 4

        if domain_size <= DOMAIN_SMALL_THRESHOLD:
            domain_index = 0
        elif domain_size <= DOMAIN_MEDIUM_THRESHOLD:
            domain_index = 1
        else:
            domain_index = 2

        return 5 * domain_index + resdis_index

    def on_negotiation_start(self, state: SAOState) -> None:
        super().on_negotiation_start(state)
        self._initialize()
        self._my_last_bid = None
        self._last_received_bid = None
        self._agent_profile = _AgentProfile()

    # ------------------------------------------------------------------
    # Bidding strategy
    # ------------------------------------------------------------------

    def _determine_next_bid(self, time: float) -> Outcome | None:
        if self._outcome_space is None or not self._outcome_space.outcomes:
            return None

        util = min(self._bs_curve.get_util(time), 1.0)

        if self._bs_use_util_range == 1:
            lower, upper = util, util + self._bs_util_range
        else:
            lower, upper = util, 1.0

        max_iterations = 2000
        for _ in range(max_iterations):
            candidates = self._outcome_space.get_bids_in_range(lower, upper)
            bid = self._oms(candidates, time)
            if bid is not None:
                return bid
            lower -= 0.001
            upper += 0.001
            if lower < self._outcome_space.min_utility - 1.0 and upper > 2.0:
                break

        return self._max_bid

    def _oms(self, candidates, time: float) -> Outcome | None:
        if len(candidates) <= 1:
            # Mirrors the original agent's quirky "keep searching" behavior.
            return None

        if self._oms_use_om == 0:
            return random.choice(candidates).bid

        util_goal = self._oms_curve.get_util(time)

        best_bid = candidates[-1].bid
        best_distance = 2.0
        for bd in candidates:
            opp_util = self._agent_profile.evaluate_bid(bd.bid)
            if opp_util > util_goal:
                continue
            distance = abs(util_goal - opp_util)
            if distance < best_distance:
                best_distance = distance
                best_bid = bd.bid
        return best_bid

    # ------------------------------------------------------------------
    # Acceptance strategy
    # ------------------------------------------------------------------

    def _accept(self, time: float) -> bool:
        if self._last_received_bid is None or self.ufun is None:
            return False

        last_opponent_util = float(self.ufun(self._last_received_bid))
        next_my_bid_util = (
            float(self.ufun(self._my_last_bid)) if self._my_last_bid is not None else 0.0
        )
        discount = self._discount_factor

        if last_opponent_util >= self._as_threshold1:
            return True
        if last_opponent_util < self._res_value * discount:
            return False

        if time <= self._as_tt1:
            return False
        elif time <= self._as_tt2:
            threshold = (
                self._as_threshold1
                - (time - self._as_tt1) * self._as_rate1
                - (1 - discount) * (time - self._as_tt1) * self._as_dr1
            )
            return last_opponent_util > threshold
        elif time <= self._as_tt3:
            threshold = (
                self._as_threshold1
                - (self._as_tt2 - self._as_tt1) * self._as_rate1 / 2
                - math.pow(10 * (time - self._as_tt2), 2) / 10 * self._as_rate2
                - (1 - discount) * (time - self._as_tt2) * self._as_dr2
            )
            return last_opponent_util > threshold
        else:
            threshold = (
                next_my_bid_util
                - (time - self._as_tt3) * self._as_rate3
                - (1 - discount) * (time - self._as_tt3) * self._as_dr3
            )
            return last_opponent_util > threshold

    # ------------------------------------------------------------------
    # SAONegotiator interface
    # ------------------------------------------------------------------

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        if not self._initialized:
            self._initialize()

        bid = self._determine_next_bid(state.relative_time)
        self._my_last_bid = bid
        return bid

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        if not self._initialized:
            self._initialize()

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        if self.ufun is None:
            return ResponseType.REJECT_OFFER

        self._last_received_bid = offer
        self._agent_profile.register_offer(offer)

        # Mirrors the Java agent's "round > 1" guard: never accept before we
        # have made at least one offer ourselves.
        if self._my_last_bid is None:
            return ResponseType.REJECT_OFFER

        if self._accept(state.relative_time):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
