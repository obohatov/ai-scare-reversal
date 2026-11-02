# AI event labels

`ai_events.csv` lists candidate AI-scare events. It holds no returns, and labels are committed to Git before any later return is viewed, so the commit date shows that the outcome did not shape them.

Label each event with:

| Column | Values | Question |
|---|---|---|
| `substitutes_paid_workflow` | yes, partly, no | Does the product replace a task that customers pay these companies for? |
| `depends_on_proprietary_data` | yes, partly, no | Would the task still need data that only these companies hold? |
| `product_status` | shipped, preview, announced, leak, none | What actually existed on the event date? |
| `label_notes` | free text | The reasoning in one or two sentences |

`named_tickers` come from the cited source. `proposed_tickers` are candidates chosen by sector and still need checking.

## Known limits

- Events enter this list because stocks fell, so the sample is selected on the outcome. It cannot show how often similar announcements caused no fall.
- Six of the eleven events fall within two weeks of early February 2026, so they are not independent.
- The February 2026 software sell-off ran across several sessions, so its events overlap.
- Late-2026 falls driven by AI spending and chip stocks are a different mechanism and are excluded.
