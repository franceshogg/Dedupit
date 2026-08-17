from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import io
import pickle
from datetime import datetime

def sort_df(df, col_order_lst, most_complete=False, nan_subset=None):

    def add_count_nan_col(df, subset):
        df['nan_count'] = df[subset].isna().sum(axis=1)
        return df
    by = []

    ascending = []

    if not nan_subset:
        nan_subset = df.columns

    if most_complete:
        df = add_count_nan_col(df, nan_subset)
        by.append('nan_count')
        ascending.append(False)

    for col, order in col_order_lst:
        by.append(col)
        ascending.append(order.lower() == 'asc')

    ret_val = df.sort_values(by=by, ascending=ascending)

    if most_complete:
        ret_val = ret_val.drop(columns='nan_count')

    ret_val['ID'] = range(1, len(ret_val) + 1)
    return ret_val


def _build_group_key(df, matching_cols):
    cols = list(matching_cols)
    key = df[cols[0]].astype(str)
    for col in cols[1:]:
        key = key + "|" + df[col].astype(str)
    return key


def _row_wildcard_compatible(norm_values, i):
    n, k = norm_values.shape
    compat = np.ones(n, dtype=bool)
    for col_idx in range(k):
        col_vals = norm_values[:, col_idx]
        vi = col_vals[i]
        is_wild_i = (vi == "__WILDCARD__")
        is_wild_col = (col_vals == "__WILDCARD__")
        same = (col_vals == vi)
        compat &= (same | is_wild_i | is_wild_col)
    return compat


def mark_duplicates(
    df, col_order_lst=None, subset=None,
    nan_cols_and_matching_cols=None, most_complete=False,
    nan_subset=None
):
    df = df.copy()
    df["dup_group"] = -1
    skipped_groups = []  # info about oversized groups skipped, surfaced to the UI

    # Sorting
    if col_order_lst is not None and len(col_order_lst) > 0:
        df = sort_df(df, col_order_lst, most_complete, nan_subset)

    if subset:
        # Drop 'obvious' duplicates (as selected by user)
        mask_complete = df[subset].notna().all(axis=1)

        df_complete = df[mask_complete]
        df_incomplete = df[~mask_complete]

        df_complete = df_complete.drop_duplicates(
            subset=subset,
            keep="first"
        )
        # Recombine
        df = pd.concat([df_complete, df_incomplete], axis=0)
        if col_order_lst:
            df = df.sort_values('ID')

    df = df.reset_index(drop=True)

    df["_orig_index"] = np.arange(len(df))

    group_id = 0

    # Marking less confident duplicates for review
    if nan_cols_and_matching_cols:
        for nan_cols, matching_cols in nan_cols_and_matching_cols:

            df_unmarked = df[df["dup_group"] == -1].copy()
            if df_unmarked.empty:
                continue

            df_unmarked["_group_key"] = _build_group_key(df_unmarked, matching_cols)

            for col in nan_cols:
                df_unmarked[col + "_norm"] = df_unmarked[col].fillna("__WILDCARD__")

            norm_cols = [col + "_norm" for col in nan_cols]

            for _, group in df_unmarked.groupby("_group_key"):
                if len(group) < 2:
                    continue

                has_null_matching = group[matching_cols].isna().any(axis=1)
                valid_group = group[~has_null_matching]

                if len(valid_group) < 2:
                    continue

                # Throws message if some groups were too large 
                if len(valid_group) > MAX_FUZZY_GROUP_SIZE:
                    sample_key = valid_group[matching_cols].iloc[0].to_dict()
                    skipped_groups.append({
                        "row_count": len(valid_group),
                        "matching_value": sample_key,
                    })
                    print(
                        f"Warning: skipping fuzzy-match group of {len(valid_group):,} rows "
                        f"sharing matching value {sample_key} - exceeds the "
                        f"{MAX_FUZZY_GROUP_SIZE:,}-row safety limit. This usually means "
                        "the matching column isn't specific enough to identify true "
                        "duplicates (too many unrelated rows share the same value)."
                    )
                    continue

                orig_indices = valid_group["_orig_index"].values
                norm_values = valid_group[norm_cols].values
                n = len(orig_indices)

                used = np.zeros(n, dtype=bool)

                for i in range(n):
                    if used[i]:
                        continue

                    current = [orig_indices[i]]
                    used[i] = True

                    row_compat = _row_wildcard_compatible(norm_values, i)

                    for j in range(i + 1, n):
                        if used[j]:
                            continue
                        if not row_compat[j]:
                            continue

                        current.append(orig_indices[j])
                        used[j] = True

                    if len(current) > 1:
                        df.loc[df["_orig_index"].isin(current), "dup_group"] = group_id
                        group_id += 1
    # Cleaning
    cols_to_drop = ["_orig_index", "_group_key"]

    if nan_cols_and_matching_cols:
        for nan_cols, _ in nan_cols_and_matching_cols:
            cols_to_drop += [col + "_norm" for col in nan_cols]

    df = df.drop(columns=cols_to_drop, errors="ignore")

    return df, skipped_groups

app = Flask(__name__)
df_store = {}  

MAX_FUZZY_GROUP_SIZE = 5000

_PERSISTENCE_AVAILABLE = True
_CURRENT_DF_PATH = None
_SKIPPED_GROUPS_PATH = None
try:
    _DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".dedup_data")
    os.makedirs(_DATA_DIR, exist_ok=True)
    _CURRENT_DF_PATH = os.path.join(_DATA_DIR, "current_df.pkl")
    _SKIPPED_GROUPS_PATH = os.path.join(_DATA_DIR, "skipped_groups.pkl")
except Exception as e:
    print(f"Warning: disk persistence unavailable, falling back to in-memory only: {e}")
    _PERSISTENCE_AVAILABLE = False


def _save_current_df(df):
    df_store["df"] = df
    if not _PERSISTENCE_AVAILABLE:
        return
    try:
        with open(_CURRENT_DF_PATH, "wb") as f:
            pickle.dump(df, f)
    except Exception as e:
        print(f"Warning: could not persist working dataframe to disk: {e}")


def _load_current_df():
    if "df" in df_store:
        return df_store["df"]
    if _PERSISTENCE_AVAILABLE and os.path.exists(_CURRENT_DF_PATH):
        try:
            with open(_CURRENT_DF_PATH, "rb") as f:
                df = pickle.load(f)
            df_store["df"] = df
            return df
        except Exception as e:
            print(f"Warning: could not load persisted dataframe: {e}")
    return None


def _save_skipped_groups(skipped_groups):
    df_store["skipped_groups"] = skipped_groups
    if not _PERSISTENCE_AVAILABLE:
        return
    try:
        with open(_SKIPPED_GROUPS_PATH, "wb") as f:
            pickle.dump(skipped_groups, f)
    except Exception as e:
        print(f"Warning: could not persist skipped-groups info to disk: {e}")


def _load_skipped_groups():
    if "skipped_groups" in df_store:
        return df_store["skipped_groups"]
    if _PERSISTENCE_AVAILABLE and _SKIPPED_GROUPS_PATH and os.path.exists(_SKIPPED_GROUPS_PATH):
        try:
            with open(_SKIPPED_GROUPS_PATH, "rb") as f:
                skipped = pickle.load(f)
            df_store["skipped_groups"] = skipped
            return skipped
        except Exception as e:
            print(f"Warning: could not load persisted skipped-groups info: {e}")
    return []


def _dataframe_to_csv_response(df, filename=None):
    if filename is None or filename.strip() == "":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"deduplicated_{timestamp}.csv"

    if not filename.lower().endswith(".csv"):
        filename += ".csv"

    safe_name = secure_filename(filename) or "deduplicated.csv"

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    csv_bytes = buffer.getvalue().encode("utf-8")

    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )

@app.route("/get_columns", methods=["POST"])
def get_columns():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        sep = "\t" if file.filename.endswith(".tsv") else ","
        df = pd.read_csv(file, sep=sep)
        df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
        columns = df.columns.tolist()

        # Store the dataframe temporarily for later use
        df_store["uploaded_df"] = df

        return jsonify({"columns": columns})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Use the stored dataframe if available
        if "uploaded_df" in df_store:
            df = df_store["uploaded_df"].copy()
        else:
            file = request.files["file"]
            sep = "\t" if file.filename.endswith(".tsv") else ","
            df = pd.read_csv(file, sep=sep)
            df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")

        # Parse col_order_lst from multiple inputs
        col_order_lst = []
        sort_col_count = int(request.form.get("sort_col_count", 0))
        for i in range(sort_col_count):
            col = request.form.get(f"sort_col_{i}")
            order = request.form.get(f"sort_order_{i}")
            if col and order:
                col_order_lst.append((col, order))

        # Parse subset (exact duplicate columns)
        subset = request.form.getlist("subset")

        # Parse nan_cols_and_matching_cols from multiple pairs
        nan_cols_and_matching_cols = []
        pair_count = int(request.form.get("pair_count", 0))
        for i in range(pair_count):
            nan_cols = request.form.getlist(f"nan_cols_{i}")
            matching_cols = request.form.getlist(f"matching_cols_{i}")

            if nan_cols and matching_cols:
                nan_cols_and_matching_cols.append((nan_cols, matching_cols))

        # Parse most_complete and nan_subset
        most_complete = request.form.get("most_complete") == "on"
        nan_subset = request.form.getlist("nan_subset") if most_complete else None

        df, skipped_groups = mark_duplicates(
            df=df,
            col_order_lst=col_order_lst if col_order_lst else None,
            subset=subset if subset else None,
            nan_cols_and_matching_cols=nan_cols_and_matching_cols if nan_cols_and_matching_cols else None,
            most_complete=most_complete,
            nan_subset=nan_subset,
        )

        _save_current_df(df)
        _save_skipped_groups(skipped_groups)

        # Clear the uploaded_df from storage
        if "uploaded_df" in df_store:
            del df_store["uploaded_df"]

        return redirect(url_for("review"))

    return render_template("index.html")

@app.route("/review", methods=["GET", "POST"])
def review():
    df = _load_current_df()
    if df is None:
        return redirect(url_for("index", session_expired=1))

    current_group_id = int(request.args.get("group_id", 0))

    dup_groups = sorted(df[df["dup_group"] != -1]["dup_group"].unique())
    total_groups = len(dup_groups)

    current_group_index = 0
    is_last_group = False

    if request.method == "POST":
        action = request.form["action"]

        if action == "save_df":
            filename = request.form.get("filename", "")
            return _dataframe_to_csv_response(df, filename)

        # GROUP REVIEW ACTIONS
        group_id = int(request.form["group_id"])
        group_rows = df[df["dup_group"] == group_id]

        if action == "delete_all":
            df.drop(index=group_rows.index[1:], inplace=True)

        elif action == "delete_selected":
            selected_indices = request.form.getlist("selected_rows")
            if selected_indices:
                selected_indices = [int(idx) for idx in selected_indices]
                df.drop(index=selected_indices, inplace=True)

        elif action == "keep_all":
            pass

        _save_current_df(df)

        next_idx = dup_groups.index(group_id) + 1
        next_group_id = dup_groups[next_idx] if next_idx < len(dup_groups) else None

        if next_group_id is not None:
            return redirect(url_for("review", group_id=next_group_id))
        else:
            return redirect(url_for("review", group_id=group_id))

    if total_groups > 0:
        if current_group_id not in dup_groups:
            current_group_id = dup_groups[0]

        current_group_index = dup_groups.index(current_group_id)
        is_last_group = current_group_index == total_groups - 1

        group_rows = df[df["dup_group"] == current_group_id]

        next_idx = current_group_index + 1
        next_group_id = dup_groups[next_idx] if next_idx < total_groups else None
    else:
        group_rows = pd.DataFrame()
        next_group_id = None
        is_last_group = True

    return render_template(
        "review.html",
        group_rows=group_rows,
        current_group_id=current_group_id,
        next_group_id=next_group_id,
        total_groups=total_groups,
        current_group_index=current_group_index,
        is_last_group=is_last_group,
        skipped_groups=_load_skipped_groups()
    )

if __name__ == "__main__":
    app.run(debug=True)