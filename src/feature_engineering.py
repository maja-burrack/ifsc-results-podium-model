import polars as pl

class Features:

    @staticmethod
    def add_features(df):
        starting_cols = df.columns
        
        df = Features._convert_data_types(df)
        
        df = df.with_columns(
            athlete_age_in_days = (pl.col('status_as_of') - pl.col('birthday')),
            athlete_years_active = pl.col('status_as_of').dt.year() - pl.col('first_season')
        )

        df = df.with_columns(is_on_podium = (pl.col('comp_rank') <= 3).cast(pl.UInt8))

        df = df.with_columns(
            jpn_athletes_count = 
                pl.col("athlete_id")
                .filter(pl.col("athlete_country") == "JPN")
                .n_unique()
                .over(partition_by=['event_id', 'dcat']),
            fra_athletes_count =
                pl.col('athlete_id')
                .filter(pl.col('athlete_country') == 'FRA')
                .n_unique()
                .over(['event_id', 'dcat'])
        )

        janja_id = 1147 

        df = df.with_columns(
            is_janja_competing =
                (pl.col('athlete_id')==janja_id)
                .any()
                .over(['event_id', 'dcat'])
                .cast(pl.UInt8)
        )

        rolling_data = (
            df.sort("status_as_of")  # important!
            .rolling(
                index_column="status_as_of",
                period="1y",
                group_by=["athlete_id"],
                closed="left"
            )
            .agg([
                pl.col("event_id").n_unique().alias("events_last_year"),
                pl.col("event_id").filter(pl.col('is_on_podium')>0).n_unique().alias('podiums_last_year'),
                (pl.col('round')=='Final').cast(pl.UInt8).sum().alias("finals_last_year"),
                (pl.col('round').str.to_lowercase() == 'semi-final').cast(pl.UInt8).sum().alias("semis_last_year"),
                pl.col('comp_rank').mean().alias('avg_rank_last_year')
            ])
        ).unique()

        df = df.join(rolling_data, on=['athlete_id', 'status_as_of'], how='left')

        df = df.with_columns(
            progression_to_semi_last_year = (pl.col('semis_last_year') / pl.col('events_last_year')).fill_nan(0),
            progression_to_final_last_year = (pl.col('finals_last_year') / pl.col('events_last_year')).fill_nan(0)
        )
        
        added_cols = list(set(df.columns).difference(set(starting_cols)))
        
        # drop first year because of the rolling windows
        min_year = df.select('status_as_of').min().item().year
        df = df.filter(pl.col('status_as_of').dt.year() > min_year)
        
        df = df.with_columns_seq(pl.col('athlete_age_in_days').dt.total_days())
        
        # all features are on a comp_level (or athlete level) so let's group data by dropping round-related columns and dropping duplicates
        cols_to_remove = [
            'round',
            'score',
            'round_rank'
        ]

        df = df.drop(cols_to_remove).unique()
        
        return df
    
    @staticmethod
    def _convert_data_types(df):
        df = df.with_columns_seq([
            pl.col("status_as_of").str.to_date("%Y-%m-%d %H:%M:%S %Z").set_sorted(),
            pl.col("birthday").str.to_date("%Y-%m-%d"),
            pl.col('first_season').cast(pl.UInt32)
        ])
        
        return df