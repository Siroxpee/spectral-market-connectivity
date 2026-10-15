"""Offline practical checks for signed electricity observations and DST."""
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd

import electricity_acquisition as e


def zipped_csv(text):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("observations.csv", text)
    return buffer.getvalue()


def must_fail(action):
    try:
        action()
    except (ValueError, KeyError, TypeError):
        return
    raise AssertionError("Invalid input unexpectedly succeeded.")


def bounds_check():
    must_fail(lambda: e.utc_bounds("2025-01-01", "2025-01-02"))
    must_fail(lambda: e.utc_bounds("2025-01-01T00:01Z", "2025-01-02T00:01Z"))
    must_fail(lambda: e.utc_bounds("2025-01-02T00:00Z", "2025-01-01T00:00Z"))
    must_fail(lambda: e.locations_checked(["A", "A"]))


def caiso_check():
    header = "INTERVALSTARTTIME_GMT,INTERVALENDTIME_GMT,NODE,LMP_TYPE,MW,MARKET_RUN_ID,OPR_DT,OPR_HR,OPR_INTERVAL\n"
    rows = "2025-01-01T00:00Z,2025-01-01T01:00Z,A,LMP,-4,DAM,2024-12-31,17,0\n2025-01-01T02:00Z,2025-01-01T03:00Z,A,LMP,0,DAM,2024-12-31,19,0\n2025-01-01T03:00Z,2025-01-01T04:00Z,A,LMP,99,DAM,2024-12-31,20,0\n"
    start, end = "2025-01-01T00:00Z", "2025-01-01T03:00Z"
    parsed = e.parse_caiso(zipped_csv(header + rows), start, end, ["A"])
    panel = e.panel_from_components(parsed, start, end, ["A"], "1h", "LMP")
    assert panel.iloc[0, 0] == -4 and np.isnan(panel.iloc[1, 0]) and panel.iloc[2, 0] == 0
    must_fail(lambda: e.parse_caiso(zipped_csv(header + rows + rows), start, end, ["A"]))
    must_fail(lambda: e.parse_caiso(zipped_csv(header + rows.replace(",-4,", ",inf,")), start, end, ["A"]))


def real_time_schema_check():
    text = "INTERVALSTARTTIME_GMT,INTERVALENDTIME_GMT,NODE,LMP_TYPE,VALUE,MARKET_RUN_ID,OPR_DT,OPR_HR,OPR_INTERVAL\n2025-01-01T00:00Z,2025-01-01T00:05Z,A,LMP,-1,RTM,2024-12-31,17,1\n"
    frame = e.parse_caiso(zipped_csv(text), "2025-01-01T00:00Z", "2025-01-01T00:05Z", ["A"], "5min", "RTM")
    assert frame.price_usd_per_mwh.tolist() == [-1]
    must_fail(lambda: e.parse_caiso(zipped_csv(text), "2025-01-01T00:00Z", "2025-01-01T01:00Z", ["A"]))


NYISO_HEADER = "Time Stamp,Name,PTID,LBMP ($/MWHr),Marginal Cost Losses ($/MWHr),Marginal Cost Congestion ($/MWHr)\n"


def dst_check():
    fall = NYISO_HEADER + "11/02/2025 00:00,A,1,-5,-1,2\n11/02/2025 01:00,A,1,0,0,0\n11/02/2025 01:00,A,1,2,1,-1\n11/02/2025 02:00,A,1,3,1,0\n"
    x = e.parse_nyiso(zipped_csv(fall), "2025-11-02T04:00Z", "2025-11-02T08:00Z", ["A"])
    p = e.panel_from_components(x, "2025-11-02T04:00Z", "2025-11-02T08:00Z", ["A"], "1h", "LBMP")
    assert p.A.tolist() == [-5, 0, 2, 3] and p.index.is_unique
    assert x.loc[x.component.eq("reported_marginal_cost_congestion"), "price_usd_per_mwh"].tolist() == [2, 0, -1, 0]
    spring = NYISO_HEADER + "03/09/2025 00:00,A,1,1,0,0\n03/09/2025 01:00,A,1,2,0,0\n03/09/2025 03:00,A,1,3,0,0\n"
    x = e.parse_nyiso(zipped_csv(spring), "2025-03-09T05:00Z", "2025-03-09T08:00Z", ["A"])
    assert x.loc[x.component.eq("LBMP"), "interval_start_utc"].diff().dropna().eq(pd.Timedelta(hours=1)).all()


def invalid_response_check():
    duplicate = NYISO_HEADER + "01/01/2025 00:00,A,1,1,0,0\n01/01/2025 00:00,A,1,1,0,0\n"
    must_fail(lambda: e.parse_nyiso(zipped_csv(duplicate), "2025-01-01T05:00Z", "2025-01-01T06:00Z", ["A"]))
    must_fail(lambda: e.zip_csvs(b"<html>gateway error</html>"))


def frozen_signed_check():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw = root / "raw.zip"; raw.write_bytes(b"source evidence")
        panel = pd.DataFrame({"A": [-5., 0., np.nan, 2.]}, index=pd.date_range("2025-01-01T00:00Z", periods=4, freq="1h"))
        panel.index.name, panel.columns.name = "interval_start_utc", "location"
        components = pd.DataFrame({"interval_start_utc": panel.index, "location": "A", "component": "LMP", "price_usd_per_mwh": panel.A.to_numpy()})
        provenance = {"units": "USD/MWh", "cache_request": e.cache_request("CAISO", panel.index[0], panel.index[-1] + pd.Timedelta(hours=1), ["A"], "1h", "DAM"), "raw_responses": [{"evidence_path": str(raw), "body_sha256": e.sha256(raw)}]}
        actual, _, target = e.freeze_observations(panel, components, provenance, root / "snapshots")
        pd.testing.assert_frame_equal(panel, actual, check_freq=False, check_exact=True)
        raw.write_bytes(b"modified")
        must_fail(lambda: e.load_observations(target))


def eia_trade_delivery_check():
    frame = pd.DataFrame({"Price hub": ["P"], "Trade date": [pd.Timestamp("2020-03-30")], "Delivery start date": [pd.Timestamp("2020-04-11")], "Delivery end date": [pd.Timestamp("2020-03-31")], "High price $/MWh": [-1.], "Low price $/MWh": [-3.], "Wtd avg price $/MWh": [-2.]})
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False)
    parsed = e.parse_eia_workbook(buffer.getvalue(), "electric", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01"), ["P"])
    assert parsed.weighted_average_price.iloc[0] == -2 and parsed.invalid_delivery_date_order.iloc[0]
    assert parsed.trade_date.iloc[0] == pd.Timestamp("2020-03-30") and parsed["Delivery start date"].iloc[0] == pd.Timestamp("2020-04-11")
    frame["Wtd avg price $/MWh"] = 10.
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False)
    bad = e.parse_eia_workbook(buffer.getvalue(), "electric", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01"), ["P"])
    assert bad.invalid_weighted_price_range.iloc[0] and bad.weighted_average_price.iloc[0] == 10.


def ercot_dst_check():
    frame = pd.DataFrame({"Delivery Date": ["11/02/2025"] * 4,
                          "Hour Ending": ["01:00", "02:00", "02:00", "03:00"],
                          "Repeated Hour Flag": ["N", "N", "Y", "N"],
                          "Settlement Point": ["A"] * 4, "Settlement Point Price": [-1., 0., 2., 3.]})
    parsed = e.parse_ercot_frame(frame, pd.Timestamp("2025-11-02T05Z"), pd.Timestamp("2025-11-02T09Z"), ["A"])
    assert parsed.interval_start_utc.tolist() == pd.date_range("2025-11-02T05Z", periods=4, freq="1h").tolist()
    assert parsed.price_usd_per_mwh.tolist() == [-1., 0., 2., 3.]
    duplicate = frame.copy(); duplicate.loc[2, "Repeated Hour Flag"] = "N"
    must_fail(lambda: e.parse_ercot_frame(duplicate, pd.Timestamp("2025-11-02T05Z"), pd.Timestamp("2025-11-02T09Z"), ["A"]))
    frame["Delivery Date"] = "03/09/2025"; frame["Hour Ending"] = ["01:00", "02:00", "04:00", "05:00"]; frame["Repeated Hour Flag"] = "N"
    spring = e.parse_ercot_frame(frame, pd.Timestamp("2025-03-09T06Z"), pd.Timestamp("2025-03-09T10Z"), ["A"])
    assert spring.interval_start_utc.diff().dropna().eq(pd.Timedelta(hours=1)).all()


def isone_dst_missing_check():
    def frame(date, labels):
        return pd.DataFrame({"Date": date, "Hr_End": labels, "DA_LMP": np.arange(len(labels), dtype=float) - 1,
                             "DA_EC": 2., "DA_CC": -1., "DA_MLC": 0.})
    fall = e.parse_isone_zone(frame("2025-11-02", ["01", "02", "02X", "03"]), pd.Timestamp("2025-11-02T04Z"), pd.Timestamp("2025-11-02T08Z"), "CT")
    total = fall.loc[fall.component.eq("DA_LMP")]
    assert total.interval_start_utc.tolist() == pd.date_range("2025-11-02T04Z", periods=4, freq="1h").tolist()
    spring = e.parse_isone_zone(frame("2025-03-09", ["01", "03", "05"]), pd.Timestamp("2025-03-09T05Z"), pd.Timestamp("2025-03-09T09Z"), "CT")
    panel = e.panel_from_components(spring, "2025-03-09T05Z", "2025-03-09T09Z", ["CT"], "1h", "DA_LMP")
    assert panel.CT.iloc[0] == -1 and panel.CT.iloc[1] == 0 and pd.isna(panel.CT.iloc[2]) and panel.CT.iloc[3] == 1
    must_fail(lambda: e.parse_isone_zone(frame("2025-03-09", ["02"]), pd.Timestamp("2025-03-09T05Z"), pd.Timestamp("2025-03-09T09Z"), "CT"))


def main():
    groups = [bounds_check, caiso_check, real_time_schema_check, dst_check, invalid_response_check, frozen_signed_check, eia_trade_delivery_check, ercot_dst_check, isone_dst_missing_check]
    for group in groups:
        group(); print("PASS", group.__name__)
    print(f"{len(groups)} offline electricity acquisition check groups passed.")


if __name__ == "__main__":
    main()
