"""
app.py
------
The Gradio user interface. All the real work happens in generator.py.
Run with:  python app.py
"""

import gradio as gr

from generator import MODELS, OUTPUT_DIR, check_providers, generate_dataset, parse_schema

# Preset use cases: name -> (purpose, schema).
# Enum options use '|' because the schema itself is split on commas.
PRESETS = {
    "Indian e-commerce support tickets": (
        "Customer support tickets from an Indian e-commerce marketplace: late deliveries, refunds, "
        "failed UPI payments, damaged products, account issues. Mix of English and Hinglish messages.",
        "ticket_id:string, customer_name:string, city:string, state:string, order_value_inr:float, "
        "category:enum(delivery|refund|payment|product_quality|account), "
        "channel:enum(email|chat|phone|whatsapp), message:string, "
        "priority:enum(low|medium|high|urgent), created_date:date(YYYY-MM-DD; between 2025-01-01 and 2026-09-25)",
    ),
    "Loan applications": (
        "Retail loan applications submitted to an Indian bank, with a realistic mix of approved "
        "and rejected applicants across income levels and employment types.",
        "applicant_id:string, age:int, city:string, "
        "employment_type:enum(salaried|self_employed|student|retired|unemployed), "
        "annual_income_inr:int, credit_score:int(300-900), "
        "loan_type:enum(home|personal|auto|education|business), loan_amount_inr:int, "
        "tenure_months:int, existing_emis:int, approved:bool",
    ),
    "Product reviews": (
        "Customer reviews of electronics, fashion and home products on an online store, "
        "ranging from glowing to furious.",
        "review_id:string, product_name:string, category:string, rating:int(1-5), "
        "review_title:string, review_text:string, verified_purchase:bool, helpful_votes:int, "
        "review_date:date(YYYY-MM-DD)",
    ),
    "Employee records": (
        "HR records for a mid-sized Indian tech company with offices in several cities.",
        "employee_id:string, full_name:string, department:string, job_title:string, "
        "hire_date:date(YYYY-MM-DD), salary_inr:int, city:string, "
        "performance_rating:float(1.0-5.0), is_remote:bool, manager_id:string",
    ),
}
FIRST_PRESET = next(iter(PRESETS))  # name of the first preset, used as the default


def provider_status():
    """Markdown summary of which providers are reachable (shown at the top of the page)."""
    status, messages = check_providers()
    lines = [
        f"{'✅' if status['gemini'] else '❌'} **Gemini**  &nbsp; "
        f"{'✅' if status['ollama'] else '❌'} **Ollama**"
    ]
    lines += [f"- ⚠️ {m}" for m in messages]
    return "\n".join(lines)


def apply_preset(preset_name):
    """Fill the purpose and schema boxes when a preset is picked."""
    if preset_name not in PRESETS:
        return gr.update(), gr.update()  # gr.update() with no arguments = "leave unchanged"
    return PRESETS[preset_name]


def run(purpose, schema_text, total_rows, rows_per_call, temperature, selected_models):
    """
    Called when the Generate button is clicked.
    gr.Error shows a clean pop-up message in the UI instead of a Python traceback.
    """
    if not purpose.strip():
        raise gr.Error("Please describe the dataset purpose.")
    if not parse_schema(schema_text):
        raise gr.Error("Please enter a schema like: name:string, age:int, city:string")
    if not selected_models:
        raise gr.Error("Please select at least one model.")

    # Check providers again on every click - Ollama may have been started/stopped since page load.
    status, messages = check_providers()
    usable = [m for m in selected_models if status[MODELS[m]]]
    skipped = [m for m in selected_models if m not in usable]

    if not usable:
        raise gr.Error("None of the selected models are reachable:\n" + "\n".join(messages))
    for message in messages:
        gr.Warning(message)  # small toast notification; generation still continues

    header = ""
    if skipped:
        header = f"Skipped (provider unavailable): {', '.join(skipped)}\n"

    # generate_dataset yields after every batch; we pass each update straight to the UI.
    for log_text, df, csv_path, stats_df in generate_dataset(
        purpose, schema_text, total_rows, rows_per_call, temperature, usable
    ):
        yield header + log_text, df, csv_path, stats_df


with gr.Blocks(title="Synthetic Data Generator") as demo:
    gr.Markdown(
        "# 🧪 Synthetic Data Generator\n"
        "Describe a dataset, and several models (Gemini + Ollama) take turns generating it in batches. "
        "Compare how the big models and the small local ones perform in the stats table."
    )
    status_md = gr.Markdown("Checking providers...")

    with gr.Row():
        # ---- Left column: inputs ----
        with gr.Column(scale=1):
            preset = gr.Dropdown(choices=list(PRESETS), value=FIRST_PRESET, label="Preset use case")
            purpose = gr.Textbox(label="Dataset purpose", lines=3, value=PRESETS[FIRST_PRESET][0])
            schema = gr.Textbox(
                label="Schema (name:type, name:type, ...)",
                lines=4,
                value=PRESETS[FIRST_PRESET][1],
                info="Use '|' inside enums, e.g. priority:enum(low|medium|high)",
            )
            total_rows = gr.Slider(10, 200, value=50, step=5, label="Total rows")
            rows_per_call = gr.Slider(5, 25, value=10, step=1, label="Rows per API call")
            temperature = gr.Slider(0.2, 1.5, value=0.9, step=0.1, label="Temperature")
            models = gr.CheckboxGroup(choices=list(MODELS), value=list(MODELS), label="Models to use")
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary")
                recheck_btn = gr.Button("Re-check providers")

        # ---- Right column: outputs ----
        with gr.Column(scale=2):
            preview = gr.Dataframe(label="Preview", wrap=True)
            csv_file = gr.File(label="Download CSV")
            stats = gr.Dataframe(label="Per-model stats")
            log_box = gr.Textbox(label="Generation log", lines=14, max_lines=30)

    # ---- Wire up events ----
    preset.change(apply_preset, inputs=preset, outputs=[purpose, schema])
    generate_btn.click(
        run,
        inputs=[purpose, schema, total_rows, rows_per_call, temperature, models],
        outputs=[log_box, preview, csv_file, stats],
    )
    recheck_btn.click(provider_status, outputs=status_md)
    demo.load(provider_status, outputs=status_md)  # runs once when the page opens


if __name__ == "__main__":
    # allowed_paths lets Gradio serve the CSV files from our outputs folder for download.
    demo.launch(allowed_paths=[str(OUTPUT_DIR)])
