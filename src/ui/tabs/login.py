"""login tab for enabling admin features via the UI."""

import gradio as gr

from src.config import AdminConfig


def build_login_tab(
    admin_indicator: gr.Markdown,
    admin_access_notice: gr.Markdown,
    admin_container: gr.Column,
    guardrail_status: gr.Markdown,
    validation_history: gr.Dataframe,
    refresh_admin_status_fn=None,
) -> None:
    """
    render login tab for activating admin tools.

    args:
        admin_indicator: global admin status indicator component
        admin_access_notice: markdown message displayed when admin tools are locked
        admin_container: column containing admin-only controls
        guardrail_status: markdown element showing system health inside the admin tab
        validation_history: dataframe listing recent validation runs
        refresh_admin_status_fn: optional callback to refresh admin panel data
    """
    with gr.Column():
        gr.Markdown(
            """
            ### Admin Login
            Authenticate to unlock the admin tools for this session.
            """
        )

        with gr.Row():
            username_input = gr.Textbox(label="Username", placeholder="Enter username")
            password_input = gr.Textbox(
                label="Password",
                type="password",
                placeholder="Enter password",
            )

        login_button = gr.Button("Sign In", variant="primary")
        logout_button = gr.Button("Sign Out", variant="secondary")
        login_status = gr.Markdown("")

        def _refresh_admin_views():
            if refresh_admin_status_fn and AdminConfig.has_access():
                try:
                    result = refresh_admin_status_fn()
                    if isinstance(result, tuple) and len(result) == 2:
                        return result
                except Exception:
                    pass
            return "", []

        def handle_login(username: str, password: str):
            success, message = AdminConfig.validate_credentials(username.strip(), password)
            indicator_text = f"**Admin Access**: {AdminConfig.status_indicator()}"

            if success:
                guardrail_text, rows = _refresh_admin_views()
                info_text = message or "Login successful. Admin features unlocked for this session."
                return (
                    f" {info_text}",
                    indicator_text,
                    gr.update(value="", visible=False),
                    gr.update(visible=True),
                    gr.update(value=guardrail_text, visible=True),
                    gr.update(value=rows, visible=True),
                )

            # failure path
            return (
                f" {message}",
                indicator_text,
                gr.update(value=AdminConfig.disabled_reason(), visible=True),
                gr.update(visible=False),
                gr.update(value="", visible=False),
                gr.update(value=[], visible=False),
            )

        def handle_logout():
            AdminConfig.revoke_session_access()
            return (
                "ℹ Session cleared. Admin features are locked.",
                f"**Admin Access**: {AdminConfig.status_indicator()}",
                gr.update(value=AdminConfig.disabled_reason(), visible=True),
                gr.update(visible=False),
                gr.update(value="", visible=False),
                gr.update(value=[], visible=False),
            )

        login_button.click(
            fn=handle_login,
            inputs=[username_input, password_input],
            outputs=[
                login_status,
                admin_indicator,
                admin_access_notice,
                admin_container,
                guardrail_status,
                validation_history,
            ],
        )

        logout_button.click(
            fn=handle_logout,
            inputs=None,
            outputs=[
                login_status,
                admin_indicator,
                admin_access_notice,
                admin_container,
                guardrail_status,
                validation_history,
            ],
        )
