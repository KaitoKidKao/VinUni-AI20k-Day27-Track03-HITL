import asyncio
import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

# Import from exercise 4 logic
from exercises.exercise_4_audit import build_graph, db_path
from common.schemas import ReviewState

# Page config
st.set_page_config(page_title="HITL PR Review Agent", layout="wide")
load_dotenv()

async def run_agent(pr_url, thread_id, action=None):
    """Run or resume the agent."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        
        if action:
            # Resume
            result = await app.ainvoke(Command(resume=action), cfg)
        else:
            # Initial run
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        
        return result

def main():
    st.title("🚀 HITL PR Review Agent")
    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.header("Cấu hình")
        pr_url = st.text_input("GitHub PR URL", value="https://github.com/VinUni-AI20k/PR-Demo/pull/1")
        
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = str(uuid.uuid4())
        
        st.info(f"Thread ID: `{st.session_state.thread_id}`")
        
        if st.button("Bắt đầu Review mới", type="primary"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.result = None
            st.session_state.last_action = None
            st.rerun()

    # Main content area
    if "result" not in st.session_state:
        st.session_state.result = None

    # Trigger run if no result yet
    if st.session_state.result is None:
        if st.button("Phân tích PR"):
            with st.spinner("Đang phân tích PR..."):
                st.session_state.result = asyncio.run(run_agent(pr_url, st.session_state.thread_id))
                st.rerun()
    
    # Handle the current state
    if st.session_state.result:
        result = st.session_state.result
        
        # Check for interrupts
        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            kind = payload.get("kind")
            
            st.subheader(f"📍 Trạng thái: Chờ phản hồi ({kind})")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.metric("Mức độ tin cậy", f"{payload.get('confidence', 0):.0%}")
                st.write(f"**Lý do:** {payload.get('confidence_reasoning')}")
                st.write(f"**Tóm tắt:** {payload.get('summary')}")
            
            with col2:
                if kind == "approval_request":
                    st.write("### Review Comments")
                    for comment in payload.get("comments", []):
                        st.warning(f"**[{comment['severity'].upper()}]** {comment['file']}:{comment.get('line') or '?'} - {comment['body']}")
                
                elif kind == "escalation":
                    st.write("### Câu hỏi cần làm rõ")
                    for q in payload.get("questions", []):
                        st.info(f"❓ {q}")

            # User input form
            st.markdown("---")
            with st.form("hitl_form"):
                if kind == "approval_request":
                    choice = st.radio("Quyết định của bạn:", ["approve", "reject", "edit"], horizontal=True)
                    feedback = st.text_area("Góp ý thêm (không bắt buộc nếu approve):")
                    
                    if st.form_submit_button("Gửi phản hồi"):
                        with st.spinner("Đang tiếp tục..."):
                            action_payload = {"choice": choice, "feedback": feedback}
                            st.session_state.result = asyncio.run(run_agent(pr_url, st.session_state.thread_id, action_payload))
                            st.rerun()
                
                elif kind == "escalation":
                    st.write("### Trả lời các câu hỏi")
                    answers = {}
                    for q in payload.get("questions", []):
                        answers[q] = st.text_input(q)
                    
                    if st.form_submit_button("Gửi câu trả lời"):
                        if all(answers.values()):
                            with st.spinner("Đang phân tích lại..."):
                                st.session_state.result = asyncio.run(run_agent(pr_url, st.session_state.thread_id, answers))
                                st.rerun()
                        else:
                            st.error("Vui lòng trả lời tất cả các câu hỏi.")
            
            # Show diff preview
            if "diff_preview" in payload:
                with st.expander("Xem trước Diff"):
                    st.code(payload["diff_preview"], language="diff")

        else:
            # Final result
            st.success(f"✅ Hoàn thành: {result.get('final_action')}")
            
            if "analysis" in result:
                a = result["analysis"]
                st.write("### Kết quả Phân tích cuối cùng")
                st.write(f"**Tóm tắt:** {a.summary}")
                st.write(f"**Mức độ tin cậy:** {a.confidence:.0%}")
                
                if a.risk_factors:
                    st.write("**Các yếu tố rủi ro:**")
                    for r in a.risk_factors:
                        st.write(f"- {r}")
                
                if a.comments:
                    st.write("**Các nhận xét:**")
                    for c in a.comments:
                        st.write(f"- **[{c.severity}]** {c.file}:{c.line or '?'} - {c.body}")

    st.markdown("---")
    st.caption("VinUni AI20k - Day 27 Track 3 - HITL Lab")

if __name__ == "__main__":
    main()
