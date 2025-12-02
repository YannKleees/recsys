import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Load Data and Model
# ---------------------------------------------------------
# Load CSVs
ratings_full = pd.read_csv("hotelrec_dense_k20_Yann.csv")
user_df = pd.read_csv("user_features.csv")
item_df = pd.read_csv("item_features.csv")

# Load Model
with open("lightfm_hybrid_model_75e_256d.pkl", "rb") as f:
    ckpt = pickle.load(f)

model = ckpt["model"]
user_id_map = ckpt["user_id_map"]
item_id_map = ckpt["item_id_map"]
inv_item_id_map = ckpt["inv_item_id_map"]
user_features_sparse = ckpt["user_features"]
item_features_sparse = ckpt["item_features"]

valid_item_ids_set = set(item_df["item_id"].values)
valid_internal_indices = np.array([
    i for i in range(item_features_sparse.shape[0])
    if inv_item_id_map[i] in valid_item_ids_set
])

# Columns to show as subratings
subrating_cols = ["rating", "sleep quality", "value", "rooms", 
                  "service", "cleanliness", "location"]

# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------
def get_top_n_recommendations(user_id, N=8):
    """Individual recommendations using ONLY valid items."""
    uidx = user_id_map[user_id]
    
    # Predict scores only for the valid subset
    scores_subset = model.predict(
        np.repeat(uidx, len(valid_internal_indices)),
        valid_internal_indices,
        user_features=user_features_sparse,
        item_features=item_features_sparse
    )

    # Map subset indices back to Item IDs
    top_subset_indices = np.argsort(-scores_subset)[:N]
    top_item_ids = [inv_item_id_map[valid_internal_indices[i]] for i in top_subset_indices]

    # Filter out items user has already rated
    user_rated = set(ratings_full[ratings_full["user_id"] == user_id]["item_id"].tolist())
    final_recs = [iid for iid in top_item_ids if iid not in user_rated]
    
    return final_recs[:N]

def get_similar_items(user_id, target_item_id, top_k=3):
    """Returns user's previously liked items similar to target."""
    user_rated = ratings_full[ratings_full["user_id"] == user_id]["item_id"].tolist()
    candidates = [iid for iid in user_rated if iid != target_item_id]
    if not candidates: return []

    target_idx = item_id_map[target_item_id]
    candidate_idx = [item_id_map[i] for i in candidates]

    target_vec = item_features_sparse[target_idx]
    candidate_vecs = item_features_sparse[candidate_idx]
    sims = cosine_similarity(target_vec, candidate_vecs)[0]
    top_idx = np.argsort(-sims)[:top_k]
    return [(candidates[i], float(sims[i])) for i in top_idx]

def compute_text_similarity(user_id, item_id):
    """Cosine similarity between user and item text embeddings."""
    emb_cols = [c for c in user_df.columns if c.startswith("text_emb_")]
    user_vec = user_df[user_df["user_id"] == user_id].iloc[0][emb_cols].to_numpy().reshape(1, -1)
    item_vec = item_df[item_df["item_id"] == item_id].iloc[0][emb_cols].to_numpy().reshape(1, -1)
    return cosine_similarity(user_vec, item_vec)[0][0]

def get_model_based_contributions(user_id, item_id):
    """Decomposes LightFM prediction into contributions."""
    uidx = user_id_map[user_id]
    iidx = item_id_map[item_id]
    _, user_embeddings = model.get_user_representations(features=user_features_sparse)
    user_vec = user_embeddings[uidx]
    item_row = item_features_sparse[iidx]
    
    contributions = {"Guest interaction patterns": 0.0, "Textual review": 0.0, "Hotel attributes": 0.0}
    meta_cols = [c for c in item_df.columns if c != "item_id"]
    n_items = len(item_id_map)

    for feature_idx in item_row.indices:
        feat_emb = model.item_embeddings[feature_idx]
        feat_bias = model.item_biases[feature_idx]
        score_part = np.dot(user_vec, feat_emb) + feat_bias
        
        if feature_idx < n_items:
            contributions["Guest interaction patterns"] += score_part
        else:
            col_idx = feature_idx - n_items 
            if col_idx < len(meta_cols):
                col_name = meta_cols[col_idx]
                if "text_emb" in col_name:
                    contributions["Textual review"] += score_part
                else:
                    contributions["Hotel attributes"] += score_part
            else:
                contributions["Hotel attributes"] += score_part

    return {k: max(0, v) for k, v in contributions.items()}

# ---------------------------------------------------------
# Group Functions
# ---------------------------------------------------------
def get_group_recommendations(group_user_ids, N=8):
    """
    Group recommendations using ONLY valid items.
    Strategy: Additive (Average).
    """
    group_scores = np.zeros((len(group_user_ids), len(valid_internal_indices)))
    visited_items = set()

    for i, uid in enumerate(group_user_ids):
        if uid not in user_id_map: continue
        
        # Predict on subset
        scores = model.predict(
            user_id_map[uid], 
            valid_internal_indices, 
            user_features=user_features_sparse, 
            item_features=item_features_sparse
        )
        group_scores[i] = scores
        user_rated = ratings_full[ratings_full["user_id"] == uid]["item_id"].tolist()
        visited_items.update(user_rated)

    # Average Scores
    agg_scores = np.mean(group_scores, axis=0)

    candidates = []
    for idx, internal_idx in enumerate(valid_internal_indices):
        iid = inv_item_id_map[internal_idx]
        if iid not in visited_items:
            candidates.append((iid, agg_scores[idx]))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in candidates[:N]]

def get_item_rank(user_id, item_id):
    """
    Calculates rank relative ONLY to the valid candidates in CSV.
    """
    if item_id not in item_id_map: return 9999

    uidx = user_id_map[user_id]
    
    scores = model.predict(
        uidx,
        valid_internal_indices,
        user_features=user_features_sparse,
        item_features=item_features_sparse
    )
    
    # Find score of target item
    target_idx = item_id_map[item_id]
    loc = np.where(valid_internal_indices == target_idx)[0]
    
    if len(loc) == 0: return 9999
    target_score = scores[loc[0]]
    
    # Rank = Count how many valid items have a higher score
    rank = np.sum(scores > target_score) + 1
    return rank

def get_member_top_reason(user_id, item_id):
    contribs = get_model_based_contributions(user_id, item_id)
    top_factor = max(contribs, key=contribs.get)
    return top_factor, contribs[top_factor]

def plot_group_recipe(group_user_ids, item_id):
    all_contribs = [get_model_based_contributions(uid, item_id) for uid in group_user_ids]
    avg_contrib = {}
    if all_contribs:
        keys = all_contribs[0].keys()
        for k in keys:
            avg_contrib[k] = sum(d[k] for d in all_contribs) / len(group_user_ids)
    total = sum(avg_contrib.values())
    if total == 0: total = 1
    contrib_norm = {k: v / total for k, v in avg_contrib.items()}
    fig, ax = plt.subplots(figsize=(7, 2))
    left = 0
    colors = ["#66c2a5", "#fc8d62", "#8da0cb"] 
    for i, (key, value) in enumerate(contrib_norm.items()):
        if value > 0:
            ax.barh("Group Score", value, left=left, color=colors[i % len(colors)], label=key, height=0.6)
            if value > 0.10: 
                ax.text(left + (value/2), 0, f"{value:.0%}", ha='center', va='center', color='white', fontweight='bold')
            left += value
    ax.axis('off')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    return fig

def interpret_contribution(contrib_dict):
    dominant = max(contrib_dict, key=contrib_dict.get)
    if dominant == "Guest interaction patterns":
        return "Fits well with the **booking habits** and history of this user."
    elif dominant == "Textual review":
        return "The **written reviews** align with what this user usually writes positively about."
    elif dominant == "Hotel attributes":
        return "The specific **amenities and ratings** align perfectly with what this user values."
    else:
        return "Balanced recommendation."


def plot_recipe(user_id, item_id):
    contrib = get_model_based_contributions(user_id, item_id)
    total = sum(contrib.values())
    if total == 0: total = 1
    contrib_norm = {k: v / total for k, v in contrib.items()}
    fig, ax = plt.subplots(figsize=(7, 2))
    left = 0
    colors = ["#66c2a5", "#fc8d62", "#8da0cb"] 
    for i, (key, value) in enumerate(contrib_norm.items()):
        if value > 0:
            ax.barh("Score", value, left=left, color=colors[i % len(colors)], label=key, height=0.6)
            if value > 0.10: 
                ax.text(left + (value/2), 0, f"{value:.0%}", ha='center', va='center', color='white', fontweight='bold')
            left += value
    ax.set_xlim(0, 1)
    ax.axis('off')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    return fig

def generate_explanation(user_id, item_id):
    item_row = item_df[item_df["item_id"] == item_id].iloc[0]
    similar = get_similar_items(user_id, item_id)
    text_sim = compute_text_similarity(user_id, item_id)
    expl = [f"### Why Hotel {item_id}?"]
    expl.append("We recommended this because it aligns with your history.")
    
    if text_sim > 0.25:
        expl.append(f"### Review Match\nThis hotel's reviews use language very similar to your past positive reviews (Similarity score: **{text_sim:.2f}**).")
    elif text_sim < -0.1:
        expl.append(f"### Something New\nGuest reviews differ from what you've seen before.")
    expl.append("### Similar hotels you liked")
    if similar:
        for sid, ssim in similar:
            expl.append(f"- Hotel **{sid}** (similarity {ssim:.2f})")
    else:
        expl.append("_No similar previously visited hotels were found._")

    expl.append("### Hotel Ratings")
    for col in subrating_cols:
        expl.append(f"- **{col.capitalize()}**: **{item_row[col]*5:.2f}**")
    
    return "\n".join(expl)

# ---------------------------------------------------------
# Streamlit App Layout
# ---------------------------------------------------------
st.title("🏨 Hotel Recommendation Explorer")

mode = st.sidebar.radio("Mode", ["Individual", "Group"])
all_users = sorted(user_df["user_id"].unique().tolist())

recommended_hotels = []
user_id = None 
group_users = []

st.sidebar.caption(f"Database: {len(valid_internal_indices)} active hotels")

if mode == "Individual":
    user_id = st.sidebar.selectbox("Select a user", all_users)
    st.sidebar.write("User selected:", user_id)
    recommended_hotels = get_top_n_recommendations(user_id, N=8)
else:
    group_users = st.sidebar.multiselect("Select Group Members", all_users, default=all_users[:3])
    if group_users:
        recommended_hotels = get_group_recommendations(group_users, N=8)

st.header("Your Recommended Hotels")

for item_id in recommended_hotels:
    item_row = item_df[item_df["item_id"] == item_id].iloc[0]
    
    with st.container(border=True):
        st.subheader(f"Hotel {item_id}")
        st.metric("Overall Rating", f"{item_row['rating'] * 5:.2f}")    

        # --- Attributes Display ---
        top_attributes = []
        if mode == "Individual":
            user_row = user_df[user_df["user_id"] == user_id].iloc[0]
            corr_cols = [c for c in user_df.columns if c.startswith("corr_rating_")]
            top_corr = user_row[corr_cols].dropna().sort_values(ascending=False).head(3)
            top_attributes = [c.replace("corr_rating_", "") for c in top_corr.index]
            if top_attributes:
                st.write(f"You seem to care most about **{', '.join(top_attributes).title()}**"
                         ". Here’s how this hotel performs on these aspects:")
        else:
            group_data = user_df[user_df["user_id"].isin(group_users)]
            corr_cols = [c for c in user_df.columns if c.startswith("corr_rating_")]
            avg_importance = group_data[corr_cols].mean().sort_values(ascending=False).head(3)
            top_attributes = [c.replace("corr_rating_", "") for c in avg_importance.index]
            st.write(f"**Group Priorities:** High average interest in **{', '.join(top_attributes).title()}**")

        if top_attributes:
            cols = st.columns(len(top_attributes))
            for idx, attr in enumerate(top_attributes):
                if attr in item_row:
                    cols[idx].metric(attr.title(), f"{item_row[attr]*5:.1f}")
        else:
            st.write("Hotel Highlights:")
            fcols = st.columns(3)
            fcols[0].metric("Cleanliness", f"{item_row['cleanliness']*5:.1f}")
            fcols[1].metric("Service", f"{item_row['service']*5:.1f}")
            fcols[2].metric("Value", f"{item_row['value']*5:.1f}")

        # --- Explanations ---
        with st.expander("See why this hotel was recommended"):
            if mode == "Individual":
                expl_text = generate_explanation(user_id, item_id)
                st.markdown(expl_text)
                st.divider()
                st.subheader("Recommendation Recipe")
                st.write("How much each factor contributed to the recommendation:")
                st.pyplot(plot_recipe(user_id, item_id), use_container_width=True)
            else:
                # Group Tabs
                tab1, tab2 = st.tabs(["👥 Group Consensus", "👤 Individual Drivers"])
                
                with tab1:
                    
                    # Group Recipe
                    st.write("The chart below shows the average factors driving this recommendation.")
                    st.pyplot(plot_group_recipe(group_users, item_id), use_container_width=True)

                    # Alignment Table
                    st.subheader("Group Alignment Map")
                    st.write(f"How well this hotel fits each person's individual taste:")
                    
                    rank_data = []
                    for uid in group_users:
                        rank = get_item_rank(uid, item_id)
                        total_items = len(valid_internal_indices)
                        percentile = (rank / total_items) * 100
                        
                        if percentile <= 5: status = "🟢 Top 5%"
                        elif percentile <= 15: status = "🟢 Top 15%"
                        elif percentile <= 30: status = "🟡 Top 30%"
                        else: status = "🔴 Compromise"
                        
                        rank_data.append({"User": uid, "Compatibility": status})
                    
                    st.dataframe(pd.DataFrame(rank_data), use_container_width=True, hide_index=True)

                with tab2:
                    st.write("Select a member to see their specific data.")
                    selected_member = st.selectbox(f"Inspect Member (Hotel {item_id})", group_users, key=f"sel_{item_id}")
                    
                    if selected_member:
                        # Member Specific Attributes
                        user_row = user_df[user_df["user_id"] == selected_member].iloc[0]
                        corr_cols = [c for c in user_df.columns if c.startswith("corr_rating_")]
                        top_corr = user_row[corr_cols].dropna().sort_values(ascending=False).head(3)
                        top_attrs = [c.replace("corr_rating_", "") for c in top_corr.index]
                        
                        if top_attrs:
                            st.write(f"**{selected_member}'s Priorities:** {', '.join(top_attrs).title()}")
                            # Show scores for these specific attributes if available
                            acols = st.columns(len(top_attrs))
                            for i, attr in enumerate(top_attrs):
                                if attr in item_row:
                                    acols[i].metric(attr.title(), f"{item_row[attr]*5:.1f}")

                        # Text Similarity
                        text_sim = compute_text_similarity(selected_member, item_id)
                        if text_sim > 0.25:
                            st.info(f"**Review Match:** Reviews here align with {selected_member}'s history (Similarity: {text_sim:.2f})")
                        
                        # Contribution Analysis
                        mem_contribs = get_model_based_contributions(selected_member, item_id)
                        mem_narrative = interpret_contribution(mem_contribs)
                        st.write(f"**Model Logic:** {mem_narrative}")
                        st.pyplot(plot_recipe(selected_member, item_id), use_container_width=True)