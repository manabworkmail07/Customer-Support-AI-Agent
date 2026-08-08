import asyncio
import warnings

from dotenv import load_dotenv
from pydantic import BaseModel

# THIS IS THE CRITICAL LINE: It tells Python to read the .env file you just made.
load_dotenv()

from agents import (
    Agent,
    Runner,
    function_tool,
    GuardrailFunctionOutput,
    input_guardrail,
    WebSearchTool,
    RunContextWrapper,
    InputGuardrailTripwireTriggered,
)

warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()


# ============================================================
# ORDERS DATABASE
# ============================================================

ORDERS_DB = {
    "ORD-001": {
        "item": "Wireless Headphones",
        "status": "Shipped",
        "eta": "March 22",
    },
    "ORD-002": {
        "item": "Python Programming Book",
        "status": "Delivered",
        "eta": "March 18",
    },
    "ORD-003": {
        "item": "USB-C Cable 3-pack",
        "status": "Processing",
        "eta": "March 25",
    },
}


# ============================================================
# ORDER LOOKUP TOOL
# ============================================================

@function_tool
def lookup_order(order_id: str) -> str:
    """Look up an order using its order ID."""

    order = ORDERS_DB.get(order_id.upper())

    if order:
        return (
            f"Order {order_id.upper()}:\n"
            f"  Item: {order['item']}\n"
            f"  Status: {order['status']}\n"
            f"  Estimated Arrival: {order['eta']}\n"
        )

    return (
        f"Order {order_id} not found. "
        "Please check the order ID and try again."
    )


# ============================================================
# REFUND TOOL
# ============================================================

@function_tool
def process_refund(order_id: str, reason: str) -> str:
    """Process a refund request for an order."""

    order = ORDERS_DB.get(order_id.upper())

    if not order:
        return f"Cannot process refund: Order {order_id} not found."

    if order["status"] == "Processing":
        return (
            f"Refund for {order_id.upper()} cannot be processed - "
            "the order hasn't shipped yet. It can be cancelled instead."
        )

    return (
        f"Refund initiated for Order {order_id.upper()}\n"
        f"Item: {order['item']}\n"
        f"Reason: {reason}\n"
        "Refund amount will be credited within 5-7 business days."
    )


# ============================================================
# SUPPORT CHECK / GUARDRAIL
# ============================================================

class SupportCheck(BaseModel):
    is_support_question: bool
    reasoning: str


guardrail_checker = Agent(
    name="Support Topic Checker",
    instructions="""Determine if the user's message is a customer support question.

Valid topics:
- Order status
- Refunds
- Returns
- Product questions
- Shipping
- Delivery
- FAQs

Invalid topics:
- Personal advice
- Jokes
- Coding help
- Unrelated conversations

Return is_support_question = True ONLY for customer support topics.""",
    output_type=SupportCheck,
)


@input_guardrail
async def support_only(
    ctx: RunContextWrapper,
    agent: Agent,
    input: str,
) -> GuardrailFunctionOutput:
    """Only allow customer support questions."""

    result = await Runner.run(
        guardrail_checker,
        input,
        context=ctx.context,
    )

    final = result.final_output

    return GuardrailFunctionOutput(
        output_info={
            "reasoning": final.reasoning,
        },
        tripwire_triggered=not final.is_support_question,
    )


# ============================================================
# ORDER STATUS AGENT
# ============================================================

order_agent = Agent(
    name="Order_Status_Agent",
    handoff_description=(
        "Handles questions about order status, shipping, and delivery."
    ),
    instructions="""You help customers check their order status.

Use the lookup_order tool to find order information.

If the customer doesn't provide an order ID, ask for it.

Be friendly and professional.""",
    tools=[lookup_order],
)


# ============================================================
# REFUND AGENT
# ============================================================

refund_agent = Agent(
    name="Refund_Agent",
    handoff_description=(
        "Handles refund requests, returns, and cancellations."
    ),
    instructions="""You help customers with refunds and returns.

Use the process_refund tool to initiate refunds.

Always ask for the order ID and reason before processing.

Be empathetic and helpful.""",
    tools=[process_refund],
)


# ============================================================
# FAQ AGENT
# ============================================================

faq_agent = Agent(
    name="FAQ_Agent",
    handoff_description=(
        "Handles general product questions and frequently asked questions."
    ),
    instructions="""You answer general customer questions and FAQs.

Use web search when you need current information.

Common topics:
- Shipping policies
- Return windows
- Product details

Be helpful and concise.""",
    tools=[WebSearchTool()],
)


# ============================================================
# TRIAGE AGENT
# ============================================================

triage_agent = Agent(
    name="Customer_Support_Triage",
    instructions="""You are the front-line customer support agent.

Your job is to understand the customer's issue and route them
to the right specialist:

- Order status, shipping, delivery questions → Order Status Agent
- Refund requests, returns, cancellations → Refund Agent
- General questions, product info, FAQs → FAQ Agent

Be warm, professional, and route quickly.""",

    handoffs=[
        order_agent,
        refund_agent,
        faq_agent,
    ],

    input_guardrails=[
        support_only
    ],
)


# ============================================================
# HANDLE CUSTOMER
# ============================================================

async def handle_customer(message: str):
    """Process a customer message through the support system."""

    print(f"👤 Customer: {message}")

    try:
        result = await Runner.run(
            triage_agent,
            message,
        )

        print(
            f"🤖 {result.last_agent.name}: "
            f"{result.final_output}"
        )

    except InputGuardrailTripwireTriggered:
        print(
            "🚫 Blocked: This doesn't appear to be "
            "a customer support question."
        )

    except Exception as e:
        print(f"❌ Error: {e}")

    print("=" * 70)
    print()


# ============================================================
# MAIN DEMO
# ============================================================

async def main():

    print("=" * 70)
    print("   CUSTOMER SUPPORT AGENT SYSTEM – DEMO")
    print("=" * 70)
    print()

    # Test 1: Order status
    # Triage → Order Status Agent → lookup_order
    await handle_customer(
        "Where is my order ORD-001?"
    )

    # Test 2: Refund request
    # Triage → Refund Agent → process_refund
    await handle_customer(
        "I want a refund for order ORD-002. "
        "The book arrived damaged."
    )

    # Test 3: General FAQ
    # Triage → FAQ Agent → Web Search
    await handle_customer(
        "What is Amazon return policy?"
    )

    # Test 4: Off-topic
    # Should be blocked by the guardrail
    await handle_customer(
        "Can you help me learn Python programming?"
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())