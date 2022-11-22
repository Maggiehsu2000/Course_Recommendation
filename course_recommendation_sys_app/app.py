# Core Pkg
import streamlit as st 
import streamlit.components.v1 as stc 
import plotly.express as px

# Load EDA
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel


# Load Our Dataset
def load_data(data):
	df = pd.read_csv(data)
	return df 


# Fxn
# Vectorize + Cosine Similarity Matrix

def vectorize_text_to_cosine_mat(data):
	count_vect = CountVectorizer()
	cv_mat = count_vect.fit_transform(data)
	# Get the cosine
	cosine_sim_mat = cosine_similarity(cv_mat)
	return cosine_sim_mat



# Recommendation Sys
@st.cache
def get_recommendation(title,cosine_sim_mat,df,num_of_rec=10):
	# indices of the course
	course_indices = pd.Series(df.index,index=df['course_title']).drop_duplicates()
	# Index of course
	idx = course_indices[title]

	# Look into the cosine matr for that index
	sim_scores =list(enumerate(cosine_sim_mat[idx]))
	sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
	selected_course_indices = [i[0] for i in sim_scores[1:]]
	selected_course_scores = [i[0] for i in sim_scores[1:]]

	# Get the dataframe & title
	result_df = df.iloc[selected_course_indices]
	result_df['similarity_score'] = selected_course_scores
	final_recommended_courses = result_df[['course_title','similarity_score','url','price','num_subscribers']]
	return final_recommended_courses.head(num_of_rec)


RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Source:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">💲Price:</span>{}</p>
<p style="color:blue;"><span style="color:black;">👨🏽‍🎓 Student:</span>{}</p>

</div>
"""

# Search For Course 
@st.cache
def search_term_if_not_found(term,df):
	result_df = df[df['course_title'].str.contains(term)]
	return result_df


def main():

	st.title("課程推薦系統")

	menu = ["Home","Recommend","Introduction","About"]
	choice = st.sidebar.selectbox("Menu",menu)

	df = load_data("data/udemy_course_data.csv")

	if choice == "Home":
		st.subheader("Home")
		st.dataframe(df.head(10))
		st.dataframe(df.info())

	elif choice == "Recommend":
		st.subheader("課程推薦")
		cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
		search_term = st.text_input("輸入課程名稱")
		num_of_rec = st.sidebar.number_input("Number",4,30,7)
		if st.button("搜尋"):
			if search_term is not None:
				try:
					results = get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)
					with st.beta_expander("Results as JSON"):
						results_json = results.to_dict('index')
						st.write(results_json)

					for row in results.iterrows():
						rec_title = row[1][0]
						rec_score = row[1][1]
						rec_url = row[1][2]
						rec_price = row[1][3]
						rec_num_sub = row[1][4]

						# st.write("Title",rec_title,)
						stc.html(RESULT_TEMP.format(rec_title,rec_score,rec_url,rec_url,rec_num_sub),height=350)
				except:
					results= "未查詢到此課程"
					st.warning(results)
					st.info("Suggested Options include")
					result_df = search_term_if_not_found(search_term,df)
					st.dataframe(result_df)

	elif choice == "Introduction":
		st.subheader("課程簡介推薦")
		cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
		search_term = st.text_input("輸入課程相關簡介")
		num_of_rec = st.sidebar.number_input("Number",4,30,7)
		if st.button("搜尋"):
			if search_term is not None:
				try:
					results = get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)
					with st.beta_expander("Results as JSON"):
						results_json = results.to_dict('index')
						st.write(results_json)

					for row in results.iterrows():
						rec_title = row[1][0]
						rec_score = row[1][1]
						rec_url = row[1][2]
						rec_price = row[1][3]
						rec_num_sub = row[1][4]

						# st.write("Title",rec_title,)
						stc.html(RESULT_TEMP.format(rec_title,rec_score,rec_url,rec_url,rec_num_sub),height=350)
				except:
					results= "未查詢到此課程"
					st.warning(results)
					st.info("Suggested Options include")
					result_df = search_term_if_not_found(search_term,df)
					st.dataframe(result_df)


				# How To Maximize Your Profits Options Trading

	else:
		st.subheader("About")
		st.text("Built with Streamlit & Pandas")


	# 在sidebar生成选择直方或者饼图的选项
		
		st.sidebar.markdown("### 課程主題分類")	#小標題
		select = st.sidebar.selectbox("視覺化類型", ["長條圖", "圓餅圖"],
									key = "1")	#下拉選單
		# 整理数据
		sentiment_count = df["subject"].value_counts() 	#value_counts=多少个不同值
		sentiment_count = pd.DataFrame({"Sentiment":sentiment_count.index,
										"Tweets":sentiment_count.values})

		# 如果没有勾选"Hide"则显示图表
		if not st.sidebar.checkbox("隱藏圖型", True):
			st.markdown('### 課程主題分類')
			if select == "長條圖":
				fig = px.bar(sentiment_count, x = "Sentiment", y = "Tweets", color = "Tweets", height = 500)
				st.plotly_chart(fig)
			else:
				fig = px.pie(sentiment_count, values = "Tweets", names = "Sentiment")
				st.plotly_chart(fig)

if __name__ == '__main__':
	main()