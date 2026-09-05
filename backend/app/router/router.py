from app.models.registry import ModelRegistry
from app.router.analyzer import QueryAnalyzer
from app.schemas.routing import RoutingDecision, QueryAnalysis


class Router:
    """Select an available model using hybrid routing: classifier + optional LLM escalation."""

    def __init__(
        self, 
        registry: ModelRegistry, 
        analyzer: QueryAnalyzer,
        classifier_confidence_threshold: float = 0.75,
    ) -> None:
        self._registry = registry
        self._analyzer = analyzer
        self._classifier_confidence_threshold = classifier_confidence_threshold

    async def route(self, query: str) -> RoutingDecision:
        try:
            classifier_tier, classifier_confidence = self._analyzer.classify(query)
        except Exception as classifier_error:
            classifier_tier = None
            classifier_confidence = 0.0

        if classifier_tier is None or classifier_confidence < self._classifier_confidence_threshold:
            try:
                predicted_tier = await self._get_llm_tier_prediction(query)
                selected_model = self._registry.get(predicted_tier)
                confidence = 0.90
                analysis = self._decision_analysis(
                    tier=predicted_tier,
                    reasoning="Selected tier using LLM escalation.",
                )
                routing_reason = (
                    f"Selected {selected_model.metadata.name} after LLM escalation."
                )
            except Exception as error:
                analysis = self._analyzer.analyze(query)
                selected_model = self._select_model(
                    required_capabilities=analysis.required_capabilities,
                    complexity=analysis.complexity,
                )
                confidence = self._calculate_confidence(
                    selected_model=selected_model.metadata.name,
                    complexity=analysis.complexity,
                )
                routing_reason = (
                    f"LLM escalation failed ({error}); selected "
                    f"{selected_model.metadata.name} using rule-based complexity fallback."
                )
        else:
            selected_model = self._registry.get(classifier_tier)
            confidence = classifier_confidence
            analysis = self._decision_analysis(
                tier=classifier_tier,
                reasoning="Selected tier using classifier decision.",
            )
            routing_reason = (
                f"Selected {selected_model.metadata.name} from the classifier decision."
            )

        return RoutingDecision(
            selected_model=selected_model.metadata.name,
            confidence=confidence,
            analysis=analysis,
            reasoning=(
                routing_reason
            ),
        )

    def _decision_analysis(self, tier: str, reasoning: str) -> QueryAnalysis:
        task_type = self._analyzer._tier_to_task_type(tier)
        return QueryAnalysis(
            task_type=task_type,
            complexity=0.0,
            required_capabilities=[],
            reasoning=reasoning,
        )

    def _select_model(
        self,
        required_capabilities: list[str],
        complexity: float,
    ):
        candidates = [
            model
            for model in self._registry.list_models()
            if set(required_capabilities).issubset(model.metadata.capabilities)
        ]

        if not candidates:
            raise ValueError(
                "No registered model supports the required capabilities: "
                f"{', '.join(required_capabilities)}"
            )

        tier_rank = {"fast": 0, "general": 1, "reasoning": 2}

        minimum_rank = 0
        if complexity >= 0.65:
            minimum_rank = 2
        elif complexity >= 0.40:
            minimum_rank = 1

        eligible_models = [
            model
            for model in candidates
            if tier_rank[model.metadata.tier] >= minimum_rank
        ]

        if not eligible_models:
            eligible_models = candidates

        return min(
            eligible_models,
            key=lambda model: (
                tier_rank[model.metadata.tier],
                model.metadata.expected_latency_ms,
            ),
        )

    @staticmethod
    def _calculate_confidence(selected_model: str, complexity: float) -> float:
        if selected_model == "reasoning":
            return 0.90

        if selected_model == "general":
            return 0.85 if complexity < 0.65 else 0.75

        return 0.90 if complexity < 0.40 else 0.70

    async def _get_llm_tier_prediction(self, query: str) -> str:
        """Use reasoning model to predict the correct tier for uncertain queries."""
        reasoning_model = self._registry.get("reasoning")
        
        prompt = (
            f"Classify this query into ONE of these tiers based on its difficulty:\n"
            f"- fast: simple factual questions, summaries, extractions\n"
            f"- general: moderate coding, debugging, technical explanations\n"
            f"- reasoning: complex algorithms, system design, optimization, proofs\n\n"
            f"Query: {query}\n\n"
            f"Respond with ONLY the tier name (fast, general, or reasoning):"
        )
        
        response = await reasoning_model.generate(prompt)
        tier_text = response.content.lower().strip()
        
        # Extract tier from response
        for tier in ["fast", "general", "reasoning"]:
            if tier in tier_text:
                return tier
        
        # Fallback if parsing fails
        return "reasoning"

