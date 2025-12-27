import streamlit as st
import time
from sql_rag_agent import run_question, answer_chain, ROLE_PASSWORDS

# Page configuration
st.set_page_config(
    page_title="SQL-RAG Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin-bottom: 1rem;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def check_authentication():
    """Check if user is authenticated using session state."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.role = None
        st.session_state.login_attempts = 0
    
    return st.session_state.authenticated

def login_page():
    """Display login page."""
    st.markdown('<div class="main-header">🔍 SQL-RAG Agent</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Authentication Required")
        st.markdown("Enter your role password to access the SQL-RAG agent.")
        
        # Password input
        password = st.text_input(
            "Enter Password",
            type="password",
            key="password_input",
            help="Admin: admin123 | Support Agent: support123"
        )
        
        # Login button
        if st.button("Login", type="primary", use_container_width=True):
            if password:
                # Check which role this password belongs to
                role = None
                for role_name, role_password in ROLE_PASSWORDS.items():
                    if password == role_password:
                        role = role_name
                        break
                
                if role:
                    st.session_state.authenticated = True
                    st.session_state.role = role
                    st.session_state.login_attempts = 0
                    st.success(f"✓ Authentication successful! Role: {role.replace('_', ' ').title()}")
                    st.rerun()
                else:
                    st.session_state.login_attempts += 1
                    remaining = 3 - st.session_state.login_attempts
                    if remaining > 0:
                        st.error(f"✗ Incorrect password. {remaining} attempt(s) remaining.")
                    else:
                        st.error("✗ Maximum login attempts exceeded. Please refresh the page.")
            else:
                st.warning("Please enter a password.")
        
        # Info box
        st.markdown("---")
        with st.expander("ℹ️ Default Passwords (Development)"):
            st.markdown("""
            - **Admin**: `admin123` - Full access to all tables and fields
            - **Support Agent**: `support123` - Restricted access (no customer emails, etc.)
            
            ⚠️ **Note**: Change these passwords in production using environment variables.
            """)

def main_interface():
    """Display main SQL-RAG interface."""
    # Header
    role_display = st.session_state.role.replace('_', ' ').title()
    st.markdown(f'<div class="main-header">🔍 SQL-RAG Agent</div>', unsafe_allow_html=True)
    
    # Role info and logout
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"👤 Logged in as: **{role_display}**")
    with col2:
        if st.button("Logout", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.role = None
            st.rerun()
    
    st.markdown("---")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Ask a Question")
        question = st.text_area(
            "Enter your question about the database:",
            height=100,
            placeholder="e.g., Which 5 customers spent the most money?",
            key="question_input"
        )
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            submit_button = st.button("🔍 Query", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 💡 Example Questions")
        examples = [
            "Which 5 customers spent the most money?",
            "What are the top 10 best-selling tracks?",
            "List all albums by AC/DC",
            "How many invoices were created in 2010?",
            "What is the total revenue by country?",
        ]
        for example in examples:
            if st.button(example, key=f"example_{hash(example)}", use_container_width=True):
                question = example
                # Process the example question immediately
                with st.spinner("Processing your question..."):
                    t0 = time.time()
                    sql, rows, err = run_question(example, st.session_state.role)
                    dt = time.time() - t0
                    
                    # Display results
                    if err:
                        st.markdown('<div class="error-box">', unsafe_allow_html=True)
                        st.error(f"**Error:** {err}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.caption(f"⏱️ Execution time: {dt:.2f}s")
                    else:
                        # Display SQL
                        st.markdown("### 📝 Generated SQL")
                        st.code(sql, language="sql")
                        
                        # Display raw results
                        with st.expander("📊 Raw Results", expanded=False):
                            st.text(str(rows))
                        
                        # Generate and display answer
                        st.markdown("### 💬 Answer")
                        with st.spinner("Generating answer..."):
                            final_answer = answer_chain.invoke({
                                "question": example,
                                "sql": sql,
                                "rows": rows
                            })
                            st.markdown(f"**{final_answer}**")
                        
                        st.caption(f"⏱️ Total execution time: {dt:.2f}s")
                        st.success("✓ Query executed successfully!")
    
    st.markdown("---")
    
    # Process question
    if submit_button and question:
        with st.spinner("Processing your question..."):
            t0 = time.time()
            sql, rows, err = run_question(question, st.session_state.role)
            dt = time.time() - t0
            
            # Display results
            if err:
                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                st.error(f"**Error:** {err}")
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption(f"⏱️ Execution time: {dt:.2f}s")
            else:
                # Display SQL
                st.markdown("### 📝 Generated SQL")
                st.code(sql, language="sql")
                
                # Display raw results
                with st.expander("📊 Raw Results", expanded=False):
                    st.text(str(rows))
                
                # Generate and display answer
                st.markdown("### 💬 Answer")
                with st.spinner("Generating answer..."):
                    final_answer = answer_chain.invoke({
                        "question": question,
                        "sql": sql,
                        "rows": rows
                    })
                    st.markdown(f"**{final_answer}**")
                
                st.caption(f"⏱️ Total execution time: {dt:.2f}s")
                
                # Success indicator
                st.success("✓ Query executed successfully!")
    
    elif submit_button and not question:
        st.warning("Please enter a question first.")

def main():
    """Main Streamlit app."""
    if not check_authentication():
        login_page()
    else:
        main_interface()

if __name__ == "__main__":
    main()

