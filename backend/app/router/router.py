from app.models.registry import ModelRegistry
from app.router.analyzer import QueryAnalyzer
from app.schemas.routing import RoutingDecision


class Router:
    """Select an available model using query requirements and complexity."""

    def __init__(self, registry: ModelRegistry, analyzer: QueryAnalyzer) -> None:
        self._registry = registry
        self._analyzer = analyzer

    def route(self, query: str) -> RoutingDecision:
        analysis = self._analyzer.analyze(query)
        selected_model = self._select_model(
            required_capabilities=analysis.required_capabilities,
            complexity=analysis.complexity,
        )

        confidence = self._calculate_confidence(
            selected_model=selected_model.metadata.name,
            complexity=analysis.complexity,
        )

        return RoutingDecision(
            selected_model=selected_model.metadata.name,
            confidence=confidence,
            analysis=analysis,
            reasoning=(
                f"Selected {selected_model.metadata.name} because it supports "
                f"{', '.join(analysis.required_capabilities)} at complexity "
                f"{analysis.complexity:.2f}."
            ),
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