from app.schemas.routing import QueryAnalysis, TaskType


class QueryAnalyzer:
    """Classify a query with transparent, deterministic rules."""

    def analyze(self, query: str) -> QueryAnalysis:
        normalized_query = query.lower().strip()
        task_type = self._detect_task_type(normalized_query)
        complexity = self._estimate_complexity(normalized_query, task_type)
        required_capabilities = self._get_required_capabilities(task_type, complexity)

        return QueryAnalysis(
            task_type=task_type,
            complexity=complexity,
            required_capabilities=required_capabilities,
            reasoning=(
                f"Detected {task_type.value} task using local keyword rules; "
                f"estimated complexity is {complexity:.2f}."
            ),
        )

    def _detect_task_type(self, query: str) -> TaskType:
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
            for keyword in ("solve", "prove", "logic", "reason step by step")
        ):
            return TaskType.REASONING

        if any(
            keyword in query
            for keyword in (
                "calculate",
                "equation",
                "integral",
                "derivative",
                "probability",
                "math",
            )
        ):
            return TaskType.MATHEMATICS

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