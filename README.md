# SDN QoS Paper Simulation (Research Track)

Mo-đun nay huong den cach lam "kieu nghien cuu sinh": co protocol thuc nghiem ro rang, lap nhieu lan, bao cao mean ± std, va co kiem dinh thong ke de danh gia tac dong cua feature scaling.

## Pham vi mo phong

- Bai toan:
  - `multi_class`: 7 lop.
  - `binary`: 2 lop.
- Mo hinh:
  - `DT`, `SVM`, `KNN`, `XGBoost`, `Hybrid SVM-DT`, `Hybrid KNN-SVM`.
- Scaling:
  - `none`, `standard`, `minmax`, `robust`.
- Metric:
  - `accuracy`, `f1_weighted`, `kappa`, `roc_auc_ovr`.

## Protocol thuc nghiem

Script se:

1. Chay nhieu `repeat` voi seed khac nhau.
2. Chay nhieu ty le `test_size` (mac dinh: `0.3,0.4,0.5`).
3. Tong hop ket qua theo `mean` va `std`.
4. Kiem dinh paired t-test giua moi scaling voi baseline `none`.

## Cai dat

```bash
cd sdn_qos_sim
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Cach chay

Mac dinh (10 repeats, test ratios 0.3/0.4/0.5):

```bash
python simulate_sdn_qos.py --output outputs
```

Chay day du hon (20 repeats):

```bash
python simulate_sdn_qos.py --output outputs --repeats 20 --test-sizes 0.3,0.4,0.5
```

## Tep ket qua

Trong thu muc `outputs`:

- `raw_runs.csv`: ket qua tung lan chay.
- `summary_mean_std.csv`: bang tong hop mean ± std.
- `best_per_setting.csv`: mo hinh tot nhat theo tung scenario + split ratio.
- `scaling_gain_tests.csv`: paired t-test cho scaling vs none.
- `*_line.png`: bieu do so sanh metric.

## Cach dung cho luan van/bai bao

- Dung `summary_mean_std.csv` de viet bang ket qua chinh.
- Dung `scaling_gain_tests.csv` de bao cao "cai thien co y nghia thong ke" (p < 0.05).
- Dung `best_per_setting.csv` de tom tat mo hinh toi uu theo tung kich ban.

## Ghi chu khoa hoc

- Ket qua la mo phong theo methodology, khong phai tai hien 1:1 vi khong dung bo du lieu IEEE DataPort goc.
- Neu ban co dataset that, chi can thay ham `build_dataset()` de giu nguyen toan bo khung danh gia.
# SDN_QOS_SIM
