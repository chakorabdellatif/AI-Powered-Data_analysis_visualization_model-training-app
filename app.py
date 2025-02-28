import pandas as pd
import streamlit as st
import plotly.express as px
from together import Together
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from models_reg import run_gradient_boosting, run_linear_regression, run_random_forest
from models_class import run_logistic_regression, run_random_forest_classifier, run_gradient_boosting_classifier
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse
import io
import joblib
import numpy as np

# Initialize Together AI client
client = Together(api_key=st.secrets["TOGETHER_API_KEY"])

# Function to load dataset
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None

# data analysis
def analyze_data(df):
    analysis = {}

    # Identifying missing values
    analysis["missing_values"] = df.isnull().sum().to_dict()

    #info on columns 
    analysis["columns"] = df.dtypes.to_dict()

    # Selecting only numeric columns for analysis
    numeric_df = df.select_dtypes(include=['number'])

    if not numeric_df.empty:
        # Outliers detection (using IQR)
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1

        outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum()
        analysis["outliers"] = outliers.to_dict()

        # Summarizing dataset statistics
        analysis["summary_statistics"] = numeric_df.describe().to_dict()
    else:
        analysis["outliers"] = "No numeric columns to analyze."
        analysis["summary_statistics"] = "No numeric columns to describe."

    # Detecting and removing duplicate rows
    analysis["duplicate_rows"] = df.duplicated().sum()
    
    return analysis

# data cleaning
def clean_data(df, handle_missing="drop", handle_outliers="remove"):
    # Handling missing values
    if handle_missing == "drop":
        df = df.dropna()
    elif handle_missing == "fill":
        numeric_df = df.select_dtypes(include=['number'])
        df[numeric_df.columns] = numeric_df.fillna(numeric_df.mean())

    # Handling outliers (only for numeric columns)
    if handle_outliers == "remove":
        numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
        df = df[mask]

    # Removing duplicate rows
    df = df.drop_duplicates()

    # Saving the cleaned dataframe to a CSV
    csv = df.to_csv(index=False)

    # Converting CSV to a downloadable link
    csv_io = io.StringIO(csv)
    st.download_button(
        label="Download Cleaned Data",
        data=csv_io.getvalue(),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )
    
    return df

# chart generate
def generate_chart_params(user_query, df_columns):
    """
    Use AI to convert natural language query to chart parameters
    Returns: dict with chart_type, x_column, y_column, aggregation, title
    """
    system_prompt = f"""You're a data visualization expert. Convert the user's question into chart parameters.
    Available columns: {df_columns}
    
    Respond ONLY with JSON:
    {{
        "chart_type": "line|bar|pie|scatter|histogram|box",
        "x_column": "column_name",
        "y_column": "column_name|none",
        "aggregation": "sum|avg|count|none",
        "title": "Descriptive title based on query"
    }}"""
    
    response = client.chat.completions.create(
        model="meta-llama/Llama-3-70b-chat-hf",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        max_tokens=500
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        st.error("Failed to parse AI response. Please try a different query.")
        return None

def render_ai_chart(df, chart_params):
    """Render chart based on AI-generated parameters"""
    chart_type = chart_params["chart_type"]
    x_column = chart_params["x_column"]
    y_column = chart_params["y_column"]
    
    # Handle case where 'none' is returned as y_column
    if y_column == "none":
        y_column = None  # This will prevent using 'none' as a column name

    # Handle aggregations
    if chart_params["aggregation"] != "none" and y_column is not None:
        df = df.groupby(x_column).agg({y_column: chart_params["aggregation"]}).reset_index()
    
    # Create appropriate visualization
    if chart_type == "line":
        fig = px.line(df, x=x_column, y=y_column)
    elif chart_type == "bar":
        fig = px.bar(df, x=x_column, y=y_column)
    elif chart_type == "pie":
        fig = px.pie(df, names=x_column, values=y_column)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_column, y=y_column)
    elif chart_type == "histogram":
        fig = px.histogram(df, x=x_column)
    elif chart_type == "box":
        fig = px.box(df, y=x_column)
    else:
        st.error("Unsupported chart type generated by AI")
        return None
    
    # Set the chart title
    fig.update_layout(title=chart_params["title"])
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- Enhanced Encoding & Target Selection ----------------------
def ai_identify_target(df):
    """Use LLM to identify optimal target variable for regression"""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    system_prompt = f"""Analyze this dataset structure and identify the best target variable for REGRESSION analysis:
    Numeric columns: {numeric_cols}
    
    Respond with JSON: {{"target": "column_name", "reason": "short_justification"}}"""
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Identify regression target variable"}
            ],
            max_tokens=300
        )
        result = json.loads(response.choices[0].message.content)
        if result["target"] not in numeric_cols:
            raise ValueError("Non-numeric target selected")
        return result
    except:
        return {"target": numeric_cols[-1], "reason": "Fallback to last numeric column"}

# ---------------------- Enhanced Data Processing ----------------------
def process_regression_data(df, target_col):
    """Process data for regression tasks with validation"""
    # Validate input
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Calculate correlations
    corr_matrix = df.corr()
    
    # Handle missing target in correlation matrix
    if target_col not in corr_matrix.columns:
        raise ValueError(f"Target '{target_col}' has no correlation values (constant or invalid)")
    
    target_corr = corr_matrix[target_col].drop(target_col, errors='ignore')
    
    # Select features with significant correlation
    selected_features = target_corr[abs(target_corr) >= 0.4].index.tolist()
    
    # Create new dataframe with validation
    try:
        processed_df = df[[target_col] + selected_features]
    except KeyError as e:
        raise ValueError(f"Error creating processed dataframe: {str(e)}")
    
    return processed_df, target_corr

# Add encoding functions
def encode_categorical_features(df, target_col):
    """Encode categorical features using One-Hot Encoding"""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    if len(categorical_cols) > 0:
        ct = ColumnTransformer(
            [('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)],
            remainder='passthrough'
        )
        encoded_array = ct.fit_transform(df)
        
        # Convert to dense array if sparse
        if issparse(encoded_array):
            encoded_array = encoded_array.toarray()
            
        feature_names = ct.get_feature_names_out()
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names)
        return encoded_df, ct
    return df, None

def ai_identify_classification_target(df):
    """Use LLM to identify optimal target variable for classification"""
    system_prompt = f"""Analyze this dataset structure and identify the best target variable for CLASSIFICATION:
    Columns: {df.columns.tolist()}
    
    Respond with JSON: {{"target": "column_name", "reason": "short_justification"}}"""
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Identify classification target variable"}
            ],
            max_tokens=300
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except:
        return {"target": df.columns[-1], "reason": "Fallback to last column"}

# NEW FUNCTION: Show target variable distribution
def visualize_target_distribution(df, target_col, task_type):
    """
    Create visualization of target variable distribution
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the target variable
    target_col (str): Name of the target column
    task_type (str): "Regression" or "Classification"
    
    Returns:
    plotly figure object
    """
    if task_type == "Regression":
        # For regression, show histogram
        fig = px.histogram(
            df, 
            x=target_col,
            title=f"Distribution of {target_col}",
            labels={target_col: "Value"},
            color_discrete_sequence=['#4CAF50']  # Green color
        )
        fig.update_layout(
            xaxis_title=target_col,
            yaxis_title="Frequency",
            bargap=0.1
        )
        
        # Add distribution statistics
        stats = df[target_col].describe()
        annotations = [
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"Mean: {stats['mean']:.2f}<br>Median: {stats['50%']:.2f}<br>Std Dev: {stats['std']:.2f}",
                showarrow=False,
                font=dict(
                    family="Arial",
                    size=14,
                    color="black"
                ),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )
        ]
        fig.update_layout(annotations=annotations)
        
    else:  # Classification
        # Get value counts for classification targets
        value_counts = df[target_col].value_counts().sort_index()
        
        # Generate a bar chart
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Distribution of {target_col} Classes",
            labels={"x": target_col, "y": "Count"},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(
            xaxis_title=target_col,
            yaxis_title="Count"
        )
    
    return fig

# NEW FUNCTION: Check class balance for classification
def check_class_balance(df, target_col):
    """
    Check class balance for classification tasks
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the target column
    target_col (str): Name of the target column
    
    Returns:
    tuple: (class_balance_info, imbalance_level, recommendations)
    """
    # Get class counts and percentages
    class_counts = df[target_col].value_counts()
    total_samples = len(df)
    class_percentages = (class_counts / total_samples * 100).round(2)
    
    # Create a dataframe for display
    class_balance_df = pd.DataFrame({
        'Count': class_counts,
        'Percentage (%)': class_percentages
    })
    
    # Determine imbalance level
    max_percentage = class_percentages.max()
    min_percentage = class_percentages.min()
    most_common_class = class_counts.idxmax()
    least_common_class = class_counts.idxmin()
    ratio = max_percentage / min_percentage if min_percentage > 0 else float('inf')
    
    # Determine imbalance level and recommendations
    if ratio <= 1.5:
        imbalance_level = "Low"
        recommendations = [
            "Standard classification algorithms should work well",
            "No special techniques required for handling imbalance"
        ]
    elif ratio <= 3:
        imbalance_level = "Moderate"
        recommendations = [
            "Consider using class weights in your model",
            "Evaluate with metrics like precision, recall, or F1-score instead of accuracy",
            "Cross-validation is highly recommended"
        ]
    elif ratio <= 10:
        imbalance_level = "High"
        recommendations = [
            "Use class weights or adjusted thresholds",
            "Consider oversampling minority classes (SMOTE, ADASYN)",
            "Consider undersampling majority classes",
            "Use F1-score or ROC-AUC for evaluation",
            "Consider ensemble methods"
        ]
    else:
        imbalance_level = "Extreme"
        recommendations = [
            "Use specialized techniques for imbalanced learning",
            "Apply SMOTE/ADASYN oversampling with careful cross-validation",
            "Consider anomaly detection approaches if appropriate",
            "Use precision-recall curves and F1-score for evaluation",
            "Consider custom loss functions penalizing minority class errors"
        ]
    
    # Create a summary dictionary
    balance_info = {
        'most_common_class': most_common_class,
        'least_common_class': least_common_class,
        'imbalance_ratio': ratio,
        'max_percentage': max_percentage,
        'min_percentage': min_percentage
    }
    
    return class_balance_df, imbalance_level, recommendations

def process_classification_data(df, target_col):
    """Process data for classification tasks"""
    try:
        # Check if target column exists in the dataframe
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
            
        # Encode target variable
        le = LabelEncoder()
        target_encoded = le.fit_transform(df[target_col])
        
        # Encode categorical features (excluding target)
        features_df = df.drop(columns=[target_col])
        encoded_df, _ = encode_categorical_features(features_df, target_col)
        
        # Combine features and target
        processed_df = encoded_df.copy()
        processed_df[target_col] = target_encoded
        
        # Calculate correlations (using numerical features only)
        num_cols = processed_df.select_dtypes(include=['number']).columns
        if len(num_cols) > 1:  # Need at least 2 numerical columns for correlation
            corr_matrix = processed_df[num_cols].corr()
            if target_col in corr_matrix.columns:
                target_corr = corr_matrix[target_col].drop(target_col, errors='ignore')
            else:
                # Create empty series if target not in correlation matrix
                target_corr = pd.Series(dtype='float64')
        else:
            target_corr = pd.Series(dtype='float64')
        
        return processed_df, target_corr, le
        
    except Exception as e:
        st.error(f"Error processing classification data: {str(e)}")
        raise

# ---------------------- Enhanced Main Function ----------------------
def main():
    st.title("🌟 AI-Powered Data Analysis & Visualization Agent 📊")

    # File Upload Section
    uploaded_file = st.file_uploader("📂 Upload Your Dataset (CSV or Excel)", type=['csv', 'xlsx'])

    if uploaded_file:
        df = load_data(uploaded_file)

        if df is not None:
            # Store the full dataframe in session state
            st.session_state.full_df = df
            
            # Existing Visualization Section
            st.subheader("👀 Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)

            # Show Detailed Analysis
            with st.expander("🔍 View Detailed Data Analysis Report"):
                analysis = analyze_data(df)

                st.write("### 🧑‍💻 Columns Information")
                columns_info = pd.Series(analysis["columns"])
                columns_info.name = 'Data Type'
                columns_info.index.name = 'Column Name'
                st.write(columns_info)

                st.write("### 📊 Statistical Summary")
                summary_info = pd.DataFrame(analysis["summary_statistics"])
                summary_info.index.name = 'Column Name'
                st.dataframe(summary_info)

                st.write("### 🚨 Duplicate Rows Count")
                st.write(analysis["duplicate_rows"])

                st.write("### 🛠️ Missing Values Analysis")
                st.bar_chart(pd.Series(analysis["missing_values"]))

                st.write("### 🧐 Outliers Detection")
                outliers_info = pd.Series(analysis["outliers"])
                outliers_info.name = 'Count'
                outliers_info.index.name = 'Column Name'
                st.write(outliers_info)

            # Automated Data Cleaning Section
            st.subheader("🧹 Automated Data Cleaning Tools")
            if st.button("🔧 Run Auto-Clean"):
                cleaned_df = clean_data(df)
                st.session_state.cleaned_df = cleaned_df
                st.session_state.full_df = cleaned_df  # Update the full dataframe
                st.success("✅ Data cleaned! Preview updated below ↓")
                st.dataframe(cleaned_df.head(), use_container_width=True)

            # Use cleaned dataframe if available, otherwise use original
            working_df = st.session_state.get('cleaned_df', df) if 'cleaned_df' in st.session_state else df
                
            # AI Analysis Section
            st.subheader("🤖 AI-Powered Data Visualization")
            user_query = st.text_input("💬 Ask a Question About Your Data (e.g., 'Show sales trends over time'): ")

            if user_query:
                with st.spinner("🔍 Analyzing data and generating visualization..."):
                    # Generate chart parameters using AI
                    chart_params = generate_chart_params(user_query, working_df.columns.tolist())

                    if chart_params:
                        # Validate parameters before rendering
                        required_params = ["x_column", "chart_type", "title"]
                        if not all(p in chart_params for p in required_params):
                            st.error("❌ Missing required chart parameters from AI response.")
                            return

                        if chart_params["x_column"] not in working_df.columns:
                            st.error(f"❌ Column '{chart_params['x_column']}' not found in dataset.")
                            return

                        render_ai_chart(working_df, chart_params)

            # Task type selection
            st.subheader("🎯 Target Variable Selection")
            task_type = st.radio("Select Analysis Type", 
                    ["Regression", "Classification"],
                    horizontal=True)
                
            if task_type == "Classification":
                try:
                    # Get full dataframe (which includes both numeric and categorical columns)
                    full_df = st.session_state.get('full_df', working_df)
                    
                    # Use AI to identify classification target
                    target_info = ai_identify_classification_target(full_df)
                    target = target_info["target"]
                    
                    # Validate target exists in dataframe
                    if target not in full_df.columns:
                        st.error(f"Target column '{target}' not found in dataset")
                        st.stop()
                        
                    # Display target distribution BEFORE encoding
                    st.subheader("📊 Target Variable Distribution")
                    target_fig = visualize_target_distribution(full_df, target, "Classification")
                    st.plotly_chart(target_fig, use_container_width=True)
                    
                    # NEW: Show class balance check
                    st.subheader("⚖️ Class Balance Analysis")
                    class_balance_df, imbalance_level, recommendations = check_class_balance(full_df, target)
                    
                    # Display class balance information
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(class_balance_df)
                    with col2:
                        st.metric("Imbalance Level", imbalance_level)
                        st.write("**Recommendations:**")
                        for rec in recommendations:
                            st.write(f"- {rec}")
                    
                    # Process classification data
                    processed_df, target_corr, le = process_classification_data(full_df, target)
                    
                    # Display encoding information
                    with st.expander("🔍 Encoded Target Classes"):
                        encoding_map = dict(zip(le.classes_, le.transform(le.classes_)))
                        st.json(encoding_map)
                        
                    # Show target selection
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Selected Target", target)
                    with col2:
                        st.write("**Selection Reason:**", target_info["reason"])
                        
                    # If we have correlations, show them
                    if not target_corr.empty:
                        st.subheader("📈 Feature-Target Correlations")
                        fig = px.bar(
                            x=target_corr.index,
                            y=target_corr.values,
                            labels={'x': 'Feature', 'y': 'Correlation'},
                            title=f"Correlation with {target}"
                        )
                        st.plotly_chart(fig)
                    
                    # Prepare for model training
                    X = processed_df.drop(columns=[target])
                    y = processed_df[target]
                    
                    # Data splitting section
                    st.subheader("🔪 Data Splitting Configuration")
                    test_size = st.slider(
                        "Select Test Set Percentage", 
                        min_value=0.1, 
                        max_value=0.4, 
                        value=0.2, 
                        step=0.05
                    )
                    
                    if st.button("Split Data"):
                        # Split the data and store in session state
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, 
                            test_size=test_size, 
                            random_state=42
                        )
                        st.session_state.split_data = {
                            'X_train': X_train,
                            'X_test': X_test,
                            'y_train': y_train,
                            'y_test': y_test,
                            'task_type': 'Classification'
                        }
                        st.success(f"""
                        ✅ Data Split Complete:
                        - Training set: {X_train.shape[0]} samples
                        - Test set: {X_test.shape[0]} samples
                        """)
                    
                except Exception as e:
                    st.error(f"Error in classification setup: {str(e)}")
                    st.stop()
            
            else:  # Regression
                # Filter to numeric columns only
                numeric_df = working_df.select_dtypes(include=['number'])
                
                # Validate numeric dataframe exists
                if numeric_df.empty:
                    st.error("No numeric columns available for regression analysis")
                    st.stop()
                    
                try:
                    target_info = ai_identify_target(numeric_df)
                    target = target_info["target"]
                    
                    # Final validation check
                    if target not in numeric_df.columns:
                        st.error(f"Target column '{target}' not found in numeric columns")
                        st.stop()
                    
                    # NEW: Show target distribution for regression
                    st.subheader("📊 Target Variable Distribution")
                    target_fig = visualize_target_distribution(numeric_df, target, "Regression")
                    st.plotly_chart(target_fig, use_container_width=True)
                    
                    # Process data and show correlations
                    processed_df, target_corr = process_regression_data(numeric_df, target)
                    
                    # Show target selection
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Selected Target", target)
                    with col2:
                        st.write("**Selection Reason:**", target_info["reason"])
                    
                    # Show correlations
                    st.subheader("📈 Feature-Target Correlations")
                    fig = px.bar(
                        x=target_corr.index,
                        y=target_corr.values,
                        labels={'x': 'Feature', 'y': 'Correlation'},
                        title=f"Correlation with {target}"
                    )
                    st.plotly_chart(fig)
                    
                    # Show selected features
                    st.subheader("🔍 Selected Features (|corr| ≥ 0.4)")
                    if len(processed_df.columns) > 1:
                        st.write(f"Selected {len(processed_df.columns)-1} features:")
                        st.dataframe(processed_df.head())
                        
                        # Show correlation matrix
                        st.subheader("🧮 Feature Correlation Matrix")
                        fig = px.imshow(
                            processed_df.corr(),
                            labels=dict(color="Correlation"),
                            color_continuous_scale='RdBu'
                        )
                        st.plotly_chart(fig)
                    else:
                        st.warning("No features met correlation threshold. Showing all numeric columns.")
                        st.dataframe(numeric_df.head())
                        
                    # Prepare features and target
                    if target in processed_df.columns:
                        X = processed_df.drop(columns=[target])
                        y = processed_df[target]
                        
                        # Data splitting section
                        st.subheader("🔪 Data Splitting Configuration")
                        test_size = st.slider(
                            "Select Test Set Percentage", 
                            min_value=0.1, 
                            max_value=0.4, 
                            value=0.2, 
                            step=0.05
                        )
                        
                        if st.button("Split Data"):
                            # Split the data and store in session state
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, 
                                test_size=test_size, 
                                random_state=42
                            )
                            st.session_state.split_data = {
                                'X_train': X_train,
                                'X_test': X_test,
                                'y_train': y_train,
                                'y_test': y_test,
                                'task_type': 'Regression'
                            }
                            st.success(f"""
                            ✅ Data Split Complete:
                            - Training set: {X_train.shape[0]} samples
                            - Test set: {X_test.shape[0]} samples
                            """)
                            
                except Exception as e:
                    st.error(f"Regression setup error: {str(e)}")
                    st.stop()
            
            # Model training section - ONLY show if split data exists
            if 'split_data' in st.session_state:
                st.subheader("🤖 Machine Learning Modeling")
                
                # Get task type from session
                current_task_type = st.session_state.split_data.get('task_type', 'Regression')
                
                if current_task_type == "Regression":
                    model_choice = st.selectbox("Choose Regression Algorithm:", [
                        "Gradient Boosting",
                        "Linear Regression", 
                        "Random Forest"
                    ])
                else:
                    model_choice = st.selectbox("Choose Classification Algorithm:", [
                        "Logistic Regression",
                        "Random Forest Classifier",
                        "Gradient Boosting Classifier"
                    ])
                
                # Scaling options
                st.subheader("⚖️ Scaling Options")
                scale_features = st.checkbox("Scale Features (Recommended)", value=True)
                scale_target = st.checkbox("Scale Target Variable (Recommended for regression with large values)")
                
                if st.button("Train Model"):
                    with st.spinner("Training model..."):
                        # Get split data from session state
                        split_data = st.session_state.split_data
                        X_train = split_data['X_train']
                        X_test = split_data['X_test']
                        y_train = split_data['y_train']
                        y_test = split_data['y_test']
                        
                        # Initialize scalers
                        scaler_X, scaler_y = None, None
                        
                        if scale_target:
                            # Feature scaling
                            scaler_X = StandardScaler()
                            X_train = scaler_X.fit_transform(X_train)
                            X_test = scaler_X.transform(X_test)
                            
                            if task_type == "Regression":
                                scaler_y = StandardScaler()
                                y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
                                y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
                            else :
                                y_train = split_data['y_train']
                                y_test = split_data['y_test']
                                
                        if task_type == "Regression":
                            
                            if model_choice == "Gradient Boosting":
                                model, mse, fig = run_gradient_boosting(X_train, X_test, y_train, y_test)
                            elif model_choice == "Linear Regression":
                                model, mse, fig = run_linear_regression(X_train, X_test, y_train, y_test)
                            else:
                                model, mse, fig = run_random_forest(X_train, X_test, y_train, y_test)
                        
                            # Store model in session state
                            st.session_state.trained_model = model
                            
                            # Show results
                            st.subheader("📊 Model Performance")
                            st.metric("Mean Squared Error", f"{mse:.4f}")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            if model_choice == "Logistic Regression":
                                model, acc, fig = run_logistic_regression(X_train, X_test, y_train, y_test)
                            elif model_choice == "Random Forest Classifier":
                                model, acc, fig = run_random_forest_classifier(X_train, X_test, y_train, y_test)
                            else:
                                model, acc, fig = run_gradient_boosting_classifier(X_train, X_test, y_train, y_test)
                        # Store model in session state
                            st.session_state.trained_model = model
                
                            st.subheader("📊 Model Performance")
                            st.metric("Accuracy Score", f"{acc:.4f}")
            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Download button
                        model_bytes = io.BytesIO()
                        joblib.dump(model, model_bytes)
                        st.download_button(
                            label="Download Trained Model",
                            data=model_bytes.getvalue(),
                            file_name=f"{model_choice.replace(' ', '_')}_model.joblib",
                            mime="application/octet-stream")
            else:
                st.warning("⚠️ Please split your data first before training models")

if __name__ == "__main__":
    main()