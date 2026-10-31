# moire-iris-lens

이중층 모아레에서 원자층 점결함(닫힌 조리개·면적 추가·토크)이 모아레 노드에 미치는 영향을
**레지스트리 사상의 위상**과 **중력 렌즈 사상** β = z − m/z̄ (m = C/κ)로 기술하는 연구 코드.

Topology of moiré nodes and iris (area/torque) defects in bilayers: degree density, node-charge Gauss law,
complex defect charge C, lens map β = z − (C/κ)/z̄, and the far/local partition 2πθ|C|/A_c · {cos φm, 1 − cos φm}.

- 논문 초안: https://claude.ai/artifact/48U2nj2hkbChCXcrEV6NzP (문서에서 Markdown/Word/PDF로 내보내기 가능)
- 주요 결과 그림: `pair/topo_charges.png`, `pair/area_add_torque.png`, `pair/conservation_test.png`, `pair/opp_pair.png`, `pair/twist_branches.png`

## 계산 의뢰서

- `tasks/TASK_mismatch_inclusion.md`: mismatch 모아레 고유팽창 원판 검증 (f_α 직접 측정). 작업 목록은 `tasks/jobs_P*.txt`, 실행은 `pair/incl_scan.py`로 하며 중간에 끊겨도 이어서 돌 수 있습니다.
- 체크포인트 백업: `tasks/ckpt_loop.sh`가 계산 중 20분마다 `pair/inc_*.npz`를 `claude/github-upload-d5q58c-ckpt` 브랜치(커밋 1개, 매번 덮어씀)에 올립니다. 새 컨테이너에서는 `tasks/ckpt_restore.sh` 후 같은 `incl_scan.py` 명령으로 이어서 돌립니다.

## 구조

| 폴더 | 내용 | 핵심 스크립트 |
| --- | --- | --- |
| `ridge/` | 1단계: 밀린·회전된 이중층의 벽(ridge)과 z 변위, 다층 | `relax2d.py`, `periodic.py`, `multilayer.py`, `ridge1d.py` |
| `hex/` | 벌집 레지스트리 모아레, 다선 노드(6·8·10선), 노드 추적, 카이랄성 | `hexcore.py`(레지스트리 V), `affine.py`, `track.py`, `chiral.py` |
| `nb/` | 버블·구멍·닫힌 조리개, 렌즈 해석(삼킴 반경, 패리티) | `iris*.py`, `iris_an.py`, `lens_an.py`, `gl_an.py` |
| `caus/` | 렌즈 이론: 4차식 상 방정식, 커스틱, 위상도 | `quartic.py`, `caustic.py`, `phase.py` |
| `pair/` | 결함 쌍·위상 전하·보존법칙 (논문 본체) | 아래 표 |

`pair/` 주요 파일:

| 파일 | 역할 |
| --- | --- |
| `pair.py` | 연속체 솔버(고정 경계, 실수 C). CLI: `python3 pair.py MODE d V0 [alpha] [orient] [sym\|xm\|c2]`, `PAIR_LAM` 환경변수로 λ |
| `pair2.py` | 복소 C(면적 + 토크) 지원 |
| `pair3.py` | 자유 경계 + 응력 없는 모아레 기준 + 전역 아핀 모드 고정 |
| `topo.py` | 노드 감김수, Q_hole, 벽 수 z, 가우스 법칙 검사 |
| `conserv.py`, `conserv_run.py` | 고리 전하 ΔQ(ρ)와 선형 예측 (보존법칙 검증) |
| `lenscharge.py` | 렌즈 근사에서 쌍이 없애는 노드 전하 |
| `neb.py`, `sweep*.py`, `seed*.py` | NEB 장벽, 벌리기/좁히기 스윕, 가지 시드 |
| `wall1d.py` | 1차원 벽 장력 γ(χ) |
| `topo_all.py`, `cdef_analysis.py`, `opp_analysis.py`, `zw_check.py` | 분석 → `*.json` |
| `fig*.py` | 그림 |

## 재현

```bash
pip install -r requirements.txt
cd pair
# 데이터(완화된 상태 *.npz)를 data 아카이브에서 pair/ 로 풀거나, 아래처럼 다시 계산
PAIR_LAM=-0.5 python3 pair.py none 0 0.005 90 x      # 꼬임 배경
PAIR_LAM=-0.5 python3 pair.py R 16 0.005 90 x        # 단일 결함 (x = d/2)
PAIR_LAM=-0.5 python3 pair.py pair 26 0.005 0 x      # 불일치 쌍
python3 cdef.py 0 -4 c_mm_add                         # 복소 C 단일 결함 (면적 추가)
python3 topo_all.py && python3 zw_check.py && python3 figTopo.py
python3 cdef_analysis.py && python3 figCdef.py
python3 conserv_run.py && python3 figConserv.py
python3 opp_analysis.py && python3 figOpp.py
```

계산 비용(기준: 2코어 클라우드 VM): 고정 경계 상태 하나 3–6분, 자유 경계 8–15분, NEB(11–15 이미지) 25–30분.
그리드 321×321(반폭 72, h = 0.45). 상자를 키우는 계산(남은 과제 1번, 반폭 ≥ 150)은 상태당 약 4배.

## 데이터

`.npz`(완화된 변위장)는 크기 때문에 git에서 제외했다. 논문에 쓴 174개 상태 + NEB 경로 4개를 float32로 압축한
`moire-iris-lens-data.tar`(약 160 MB)를 GitHub Release 자산이나 Zenodo로 올리고, `pair/`에 풀면 분석·그림 스크립트가 그대로 동작한다.
각 `.npz`: `x`(격자 좌표), `ux`, `uy`(상대변위), `core`(코어 마스크). 에너지는 같은 이름의 `.json`(저장소에 포함).

## 매개변수

θ = 1/30, V₀ = 0.005, V₃ = √3V₀, μ = 1, λ ∈ {−0.5, 1} (K/μ = 0.5, 2), |C| = 4, r_h = 2.4, R = √(|C|/θ) ≈ 11.
