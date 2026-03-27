"""
ComfyUI_RN_External_Interface - Billing Engine
计费引擎，支持多种计费类型的灵活扩展

支持计费类型:
- token: 按 token 计费
- per_use: 按次计费
- per_second: 按秒计费
- per_model: 按模型计费（固定费用）
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class BillingType(Enum):
    """支持的计费类型枚举"""
    TOKEN = "token"
    PER_USE = "per_use"
    PER_SECOND = "per_second"
    PER_MODEL = "per_model"
    # 预留扩展类型
    UNKNOWN = "unknown"


@dataclass
class BillingResult:
    """计费结果"""
    estimated: float = 0.0      # 预估费用 (USD)
    actual: float = 0.0         # 实际费用 (USD)
    currency: str = "USD"
    billing_type: str = "unknown"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_display_dict(self) -> Dict:
        return {
            "estimated": self.estimated,
            "actual": self.actual,
            "currency": self.currency,
            "billing_type": self.billing_type,
            "details": self.details,
            "estimated_credits": self.estimated * 211,
            "actual_credits": self.actual * 211,
        }


@dataclass
class NodeBillingRecord:
    """节点计费记录"""
    node_id: str
    node_name: str
    node_type: str
    result: BillingResult
    is_billable: bool = True


@dataclass
class WorkflowBillingSummary:
    """Workflow 计费汇总"""
    workflow_id: str = ""
    node_records: List[NodeBillingRecord] = field(default_factory=list)
    total_estimated: float = 0.0
    total_actual: float = 0.0

    def add(self, node_id: str, node_name: str, node_type: str, result: BillingResult, is_billable: bool = True):
        record = NodeBillingRecord(node_id, node_name, node_type, result, is_billable)
        self.node_records.append(record)
        if is_billable:
            self.total_estimated += result.estimated
            self.total_actual += result.actual

    def to_display_dict(self) -> Dict:
        return {
            "workflow_id": self.workflow_id,
            "total_estimated_usd": self.total_estimated,
            "total_actual_usd": self.total_actual,
            "total_estimated_credits": self.total_estimated * 211,
            "total_actual_credits": self.total_actual * 211,
            "nodes": [
                {
                    "id": r.node_id,
                    "name": r.node_name,
                    "type": r.node_type,
                    "estimated": r.result.estimated,
                    "actual": r.result.actual,
                    "is_billable": r.is_billable,
                }
                for r in self.node_records
            ]
        }


class BillingCalculator:
    """
    可扩展的计费计算器

    使用装饰器注册新的计费策略:
    @BillingCalculator.register("custom_type")
    def calc_custom(model_config, data, is_estimate):
        ...
    """

    # 计费策略注册表
    _strategies: Dict[str, Callable] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, billing_type: str) -> Callable:
        """注册新的计费策略装饰器"""
        def decorator(func: Callable) -> Callable:
            cls._strategies[billing_type] = func
            logger.debug(f"Registered billing strategy: {billing_type}")
            return func
        return decorator

    @classmethod
    def get_strategy(cls, billing_type: str) -> Optional[Callable]:
        """获取计费策略"""
        return cls._strategies.get(billing_type)

    @classmethod
    def calculate_estimate(cls, model_key: str, params: Dict) -> BillingResult:
        """
        根据模型配置和输入参数计算预估费用

        Args:
            model_key: 模型标识符 (如 "openai/gpt-4")
            params: 参数字典，包含计费所需的各项参数

        Returns:
            BillingResult: 计费结果
        """
        from comfly_config import get_billing_config

        config = get_billing_config()
        models_config = config.get("models", {})
        model_config = models_config.get(model_key)

        if not model_config:
            # 如果没有找到精确匹配，尝试模糊匹配
            for key, value in models_config.items():
                if model_key.lower().startswith(key.lower()) or key.lower() in model_key.lower():
                    model_config = value
                    break

        if not model_config:
            return BillingResult(billing_type="unknown")

        billing_type = model_config.get("billing_type", "unknown")
        strategy = cls._strategies.get(billing_type)

        if not strategy:
            logger.warning(f"No billing strategy for type: {billing_type}")
            return BillingResult(billing_type=billing_type)

        try:
            estimated = strategy(model_config, params, is_estimate=True)
            return BillingResult(
                estimated=estimated,
                billing_type=billing_type,
                details={"model": model_key, "params": params}
            )
        except Exception as e:
            logger.error(f"Error calculating estimate for {model_key}: {e}")
            return BillingResult(billing_type=billing_type)

    @classmethod
    def calculate_actual(cls, model_key: str, params: Dict, result_data: Dict) -> BillingResult:
        """
        根据模型配置、参数和执行结果计算实际费用

        Args:
            model_key: 模型标识符
            params: 输入参数
            result_data: 执行结果数据（包含实际消耗信息）

        Returns:
            BillingResult: 计费结果
        """
        from comfly_config import get_billing_config

        config = get_billing_config()
        models_config = config.get("models", {})
        model_config = models_config.get(model_key)

        if not model_config:
            for key, value in models_config.items():
                if model_key.lower().startswith(key.lower()) or key.lower() in model_key.lower():
                    model_config = value
                    break

        if not model_config:
            return BillingResult(billing_type="unknown")

        billing_type = model_config.get("billing_type", "unknown")
        strategy = cls._strategies.get(billing_type)

        if not strategy:
            return BillingResult(billing_type=billing_type)

        try:
            actual = strategy(model_config, result_data, is_estimate=False)
            return BillingResult(
                estimated=0,  # 预估在执行前已计算
                actual=actual,
                billing_type=billing_type,
                details={"model": model_key, "result": result_data}
            )
        except Exception as e:
            logger.error(f"Error calculating actual for {model_key}: {e}")
            return BillingResult(billing_type=billing_type)

    @classmethod
    def init_strategies(cls):
        """初始化内置计费策略（自动调用）"""
        if cls._initialized:
            return

        # Token 计费
        @cls.register("token")
        def calc_token(model_config: Dict, data: Dict, is_estimate: bool) -> float:
            """
            按 token 计费
            model_config 需要: input_price_per_1k, output_price_per_1k
            data 需要 (预估): estimated_input_tokens, estimated_output_tokens
            data 需要 (实际): input_tokens, output_tokens
            """
            if is_estimate:
                input_tokens = data.get("estimated_input_tokens", 0)
                output_tokens = data.get("estimated_output_tokens", 0)
            else:
                input_tokens = data.get("input_tokens", 0)
                output_tokens = data.get("output_tokens", 0)

            input_price = model_config.get("input_price_per_1k", 0)
            output_price = model_config.get("output_price_per_1k", 0)

            return (input_tokens / 1000 * input_price) + (output_tokens / 1000 * output_price)

        # 按次计费
        @cls.register("per_use")
        def calc_per_use(model_config: Dict, data: Dict, is_estimate: bool) -> float:
            """
            按次计费（固定费用）
            model_config 需要: price_per_use
            data: 次数（默认1次）
            """
            count = data.get("estimated_count", 1) if is_estimate else data.get("count", 1)
            return count * model_config.get("price_per_use", 0)

        # 按秒计费
        @cls.register("per_second")
        def calc_per_second(model_config: Dict, data: Dict, is_estimate: bool) -> float:
            """
            按秒计费
            model_config 需要: price_per_second
            data 需要 (预估): estimated_duration_seconds
            data 需要 (实际): actual_duration_seconds
            """
            if is_estimate:
                duration = data.get("estimated_duration_seconds", 0)
            else:
                duration = data.get("actual_duration_seconds", 0)

            return duration * model_config.get("price_per_second", 0)

        # 按模型计费（固定费用）
        @cls.register("per_model")
        def calc_per_model(model_config: Dict, data: Dict, is_estimate: bool) -> float:
            """
            按模型计费（每次调用固定费用）
            model_config 需要: price_per_model
            """
            return model_config.get("price_per_model", 0)

        cls._initialized = True
        logger.info("Billing strategies initialized")


# 全局 workflow 计费跟踪
_workflow_billings: Dict[str, WorkflowBillingSummary] = {}


def get_workflow_billing(workflow_id: str) -> WorkflowBillingSummary:
    """获取或创建 workflow 计费摘要"""
    if workflow_id not in _workflow_billings:
        _workflow_billings[workflow_id] = WorkflowBillingSummary(workflow_id=workflow_id)
    return _workflow_billings[workflow_id]


def reset_workflow_billing(workflow_id: str):
    """重置 workflow 计费数据"""
    if workflow_id in _workflow_billings:
        del _workflow_billings[workflow_id]


def format_price_for_display(price_usd: float) -> str:
    """
    将价格格式化为显示字符串

    Args:
        price_usd: 价格（美元）

    Returns:
        格式化的字符串，如 "$0.0023" 或 "2.3 credits"
    """
    if price_usd < 0.01:
        # 小金额显示为 credits
        credits = price_usd * 211
        credits_str = f"{credits:,.1f}".rstrip("0").rstrip(".")
        return f"{credits_str} credits"
    else:
        return f"${price_usd:.4f}"


# 初始化内置策略
BillingCalculator.init_strategies()


# ============== 便捷函数 ==============

def estimate_price(model_key: str, **kwargs) -> BillingResult:
    """
    便捷函数：估算价格

    用法:
        result = estimate_price("openai/gpt-4", estimated_input_tokens=1000, estimated_output_tokens=500)
        print(result.estimated)  # 0.04 (假设的定价)
    """
    return BillingCalculator.calculate_estimate(model_key, kwargs)


def calculate_actual_price(model_key: str, result_data: Dict) -> BillingResult:
    """
    便捷函数：计算实际价格

    用法:
        result = calculate_actual_price("openai/gpt-4", {"input_tokens": 1000, "output_tokens": 500})
        print(result.actual)
    """
    return BillingCalculator.calculate_actual(model_key, result_data, {})
