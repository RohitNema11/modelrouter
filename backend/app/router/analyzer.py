from app.schemas.routing import QueryAnalysis, TaskType
from app.router.classifier import create_default_classifier


class QueryAnalyzer:
    """Classify a query using ML classifier (primary) with keyword fallback."""

    def __init__(self, use_classifier: bool = True):
        self.use_classifier = use_classifier
        self.classifier = None
        
        if use_classifier:
            self.classifier = create_default_classifier()

    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze a query using the keyword and complexity rules."""
        normalized_query = query.lower().strip()
        task_type = self._detect_task_type_keywords(normalized_query)
        reasoning_msg = f"Detected {task_type.value} task using keyword rules."
        
        complexity = self._estimate_complexity(normalized_query, task_type)
        required_capabilities = self._get_required_capabilities(task_type, complexity)

        return QueryAnalysis(
            task_type=task_type,
            complexity=complexity,
            required_capabilities=required_capabilities,
            reasoning=reasoning_msg,
        )

    def classify(self, query: str) -> tuple[str, float]:
        """Return only the classifier's tier and confidence."""
        if not self.use_classifier or self.classifier is None:
            raise RuntimeError("Classifier is unavailable.")

        return self.classifier.predict(query.lower().strip())

    def _detect_task_type_keywords(self, query: str) -> TaskType:
        """Fallback: keyword-based detection (original logic)."""
        if any(
            keyword in query
            for keyword in ("debug", "bug", "error", "exception", "traceback")
        ):
            return TaskType.DEBUGGING

        if any(
            keyword in query
            for keyword in (
                "python",
                "javascript",
                "typescript",
                "java",
                "c++",
                "code",
                "function",
                "algorithm",
                "implement",
            )
        ):
            return TaskType.CODING

        if any(
            keyword in query
            for keyword in (
                "calculate",
                "equation",
                "integral",
                "derivative",
                "probability",
                "math",
                "fraction",
                "ratio",
                "percentage",
                "reciprocal",
                "work rate",
            )
        ):
            return TaskType.MATHEMATICS

        if any(
            keyword in query
            for keyword in (
                "solve",
                "prove",
                "logic",
                "reason step by step",
                "step by step",
                "deduce",
                "infer",
                "work through",
            )
        ):
            return TaskType.REASONING

        if any(keyword in query for keyword in ("summarize", "summary", "tl;dr")):
            return TaskType.SUMMARIZATION

        if any(keyword in query for keyword in ("extract", "find all", "parse")):
            return TaskType.EXTRACTION

        if any(
            keyword in query
            for keyword in ("write a story", "poem", "creative", "brainstorm")
        ):
            return TaskType.CREATIVE

        return TaskType.GENERAL

    def _estimate_complexity(self, query: str, task_type: TaskType) -> float:
        complexity = 0.20

        if task_type in {TaskType.CODING, TaskType.DEBUGGING, TaskType.MATHEMATICS}:
            complexity += 0.25

        if task_type == TaskType.REASONING:
            complexity += 0.35

        if len(query.split()) > 30:
            complexity += 0.15

        if any(
            phrase in query
            for phrase in (
                "optimize",
                "production",
                "distributed",
                "complex",
                "advanced",
                "time complexity",
                "edge cases",
                "step by step",
            )
        ):
            complexity += 0.20

        return min(complexity, 1.0)

    def _get_required_capabilities(
        self,
        task_type: TaskType,
        complexity: float,
    ) -> list[str]:
        capabilities_by_task = {
            TaskType.CODING: ["coding"],
            TaskType.DEBUGGING: ["debugging"],
            TaskType.MATHEMATICS: ["mathematics"],
            TaskType.SUMMARIZATION: ["summarization"],
            TaskType.EXTRACTION: ["extraction"],
            TaskType.REASONING: ["complex_reasoning"],
        }

        capabilities = capabilities_by_task.get(task_type, ["general"])

        if complexity >= 0.65 and "complex_reasoning" not in capabilities:
            capabilities.append("complex_reasoning")

        return capabilities
