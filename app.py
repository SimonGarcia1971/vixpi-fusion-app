import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

# ── Configuración de página ──────────────────────────
st.set_page_config(
    page_title="VIXπ-Fusion · Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personalizado ────────────────────────────────
st.markdown("""
<style>
  .main { background: #0d0f14; }
  [data-testid="stSidebar"] { background: #13161e; }
  .stMetric { background: #1a1e28; border-radius: 10px; padding: 12px; }
  div[data-testid="metric-container"] {
    background: #1a1e28;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 14px;
  }
  .exc-cell  { background: rgba(26,122,94,0.25) !important; }
  .evit-cell { background: rgba(180,50,50,0.15) !important; }
  h1,h2,h3 { color: #e8eaf0; }
  .stButton > button {
    background: #1a7a5e; color: white;
    border: none; border-radius: 8px;
  }
  .categoria-exc  { color: #2ecc9a; font-weight: 700; }
  .categoria-bue  { color: #52c89a; font-weight: 600; }
  .categoria-ace  { color: #dca028; }
  .categoria-evi  { color: #c04444; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# FUNCIONES DE CÁLCULO
# ════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def calcular_indicador(df_raw):
    df = df_raw.copy()
    df.rename(columns={'Unnamed: 0':'datetime'}, inplace=True)
    df = df.reset_index(drop=True)
    df['dt']        = pd.to_datetime(df['datetime'].str.replace('/', ' '))
    df['hour']      = df['dt'].dt.hour
    df['fecha']     = df['dt'].dt.date
    df['dia_num']   = df['dt'].dt.dayofweek
    df['dia_nombre']= df['dt'].dt.day_name()

    vix_length = 22
    df['vix_high'] = df['high'].shift(1).rolling(vix_length).max()
    df['vix_low']  = df['low'].shift(1).rolling(vix_length).min()
    df['vix_fix_raw'] = (df['vix_high']-df['close'].shift(1))/(df['vix_high']-df['vix_low'])*100
    df['vix_fix'] = df['vix_fix_raw'].ewm(span=3,adjust=False).mean()

    df['corr_35'] = df['low'].shift(1).rolling(35).corr(df['low'].shift(36)).fillna(0)
    df['corr_50'] = df['high'].shift(1).rolling(50).corr(df['high'].shift(51)).fillna(0)
    df['corr_60'] = df['high'].shift(1).rolling(60).corr(df['high'].shift(61)).fillna(0)
    for c in ['corr_35','corr_50','corr_60']:
        norm=(df[c]+1)/2; df[c+'_ciclo']=np.sin(np.pi*norm/2)
    corr_avg=(df['corr_35_ciclo']+df['corr_50_ciclo']+df['corr_60_ciclo'])/3

    pw,pc=0.4,0.6
    df['senal_pro']=np.sin(np.pi*df['vix_fix']/100)*pw+corr_avg*pc
    df['senal_17'] =np.sin(2*np.pi*(df['vix_fix']/100)*(1/7))*pw+corr_avg*pc
    df['ema_pro']  =df['senal_pro'].ewm(span=21,adjust=False).mean()
    df['ema_17']   =df['senal_17'].ewm(span=21,adjust=False).mean()
    df['ema17_sobre_pro']=df['ema_17']>df['ema_pro']
    df['cruce']=(df['ema_17']>df['ema_pro'])&(df['ema_17'].shift(1)<=df['ema_pro'].shift(1))

    df['tr']=np.maximum(df['high']-df['low'],
        np.maximum(abs(df['high']-df['close'].shift(1)),abs(df['low']-df['close'].shift(1))))
    df['atr14']=df['tr'].rolling(14).mean()

    df['thales_corr']=(df['corr_50']-df['corr_35'])/(df['corr_60']-df['corr_35']+0.0001)
    df['thales_corr']=df['thales_corr'].clip(-5,5)
    df['pend_pro_3']=df['ema_pro']-df['ema_pro'].shift(3)
    return df

@st.cache_data(show_spinner=False)
def calcular_grid_dia_hora(df, ventana=150, min_trades=3):
    grid_tpsl=[(8,15),(8,20),(10,15),(10,20),(10,25),(10,30),
               (12,15),(12,20),(12,25),(12,30),(12,40),(12,50),
               (15,20),(15,25),(15,30),(15,40),(15,50),
               (20,30),(20,40),(20,50)]

    signal_idxs=df[df['cruce']].index.tolist()
    DIAS=['Lunes','Martes','Miércoles','Jueves','Viernes']

    def sim(idxs,sl,tp):
        res=[]
        for idx in idxs:
            if idx+ventana>=len(df) or idx<200: continue
            entry=df.loc[idx,'close']; pnl=0
            for j in range(idx+1,min(idx+ventana,len(df))):
                if not df.loc[j,'ema17_sobre_pro']:
                    pnl=round(df.loc[j,'close']-entry,2); break
                if df.loc[j,'low']<=entry-sl: pnl=-sl; break
                if df.loc[j,'high']>=entry+tp: pnl=tp; break
            res.append(pnl)
        arr=np.array(res)
        if len(arr)==0: return None
        wins=(arr>0).sum()
        pf=arr[arr>0].sum()/abs(arr[arr<=0].sum()) if (arr<=0).sum()>0 else 99
        eq=np.cumsum(arr)
        dd=(np.maximum.accumulate(np.append([0],eq))-np.append([0],eq)).max()
        return {'n':len(arr),'wr':round(wins/len(arr)*100,1),
                'pnl':round(arr.sum(),1),'pf':round(pf,2),'dd':round(dd,1)}

    def cat(pf,wr,pnl):
        if pf>=2.0 and wr>=55: return 'EXCELENTE'
        if pf>=1.5 and wr>=50: return 'BUENO'
        if pf>=1.2 and pnl>0:  return 'ACEPTABLE'
        if pnl>0:               return 'MARGINAL'
        return 'EVITAR'

    resultados=[]
    for dia in range(5):
        for hora in range(24):
            idxs=[i for i in signal_idxs
                  if i+ventana<len(df) and i>=200
                  and df.loc[i,'dia_num']==dia
                  and df.loc[i,'hour']==hora]
            if len(idxs)<min_trades: continue
            best={'pnl':-9999,'sl':0,'tp':0,'wr':0,'pf':0,'n':0}
            for sl,tp in grid_tpsl:
                r=sim(idxs,sl,tp)
                if r and r['pnl']>best['pnl'] and r['n']>=min_trades:
                    best={**r,'sl':sl,'tp':tp}
            if best['sl']==0: continue
            night=hora<10 or hora>=20
            resultados.append({
                'dia':dia,'hora':hora,'dia_nombre':DIAS[dia],
                'n':len(idxs),
                'best_sl':best['sl'],'best_tp':best['tp'],
                'best_wr':best['wr'],'best_pnl':best['pnl'],
                'best_pf':best['pf'],'categoria':cat(best['pf'],best['wr'],best['pnl']),
                'night':night
            })
    return pd.DataFrame(resultados)

@st.cache_data(show_spinner=False)
def backtest_filtrado(df, filtro_tipo='thales_atr', sl=12, tp=20, n_contratos=2, comision=5.0):
    HORAS_OK=list(range(10,20))
    VENTANA=150
    signal_idxs=df[df['cruce']].index.tolist()

    if filtro_tipo=='base':
        filtro=lambda i: df.loc[i,'hour'] in HORAS_OK
    elif filtro_tipo=='thales':
        filtro=lambda i: (df.loc[i,'hour'] in HORAS_OK and
                         np.isfinite(df.loc[i,'thales_corr']) and
                         df.loc[i,'thales_corr']>0.5)
    elif filtro_tipo=='thales_atr':
        filtro=lambda i: (df.loc[i,'hour'] in HORAS_OK and
                         np.isfinite(df.loc[i,'thales_corr']) and
                         df.loc[i,'thales_corr']>0.5 and
                         df.loc[i,'atr14']<=14)
    else:  # pro_thales_atr
        filtro=lambda i: (df.loc[i,'hour'] in HORAS_OK and
                         df.loc[i,'pend_pro_3']<0 and
                         np.isfinite(df.loc[i,'thales_corr']) and
                         df.loc[i,'thales_corr']>0.5 and
                         df.loc[i,'atr14']<=14)

    trades=[]
    for idx in signal_idxs:
        if idx+VENTANA>=len(df) or idx<200: continue
        if not filtro(idx): continue
        entry=df.loc[idx,'close']; pnl=0
        for j in range(idx+1,min(idx+VENTANA,len(df))):
            if not df.loc[j,'ema17_sobre_pro']:
                pnl=round(df.loc[j,'close']-entry,2); break
            if df.loc[j,'low']<=entry-sl: pnl=-sl; break
            if df.loc[j,'high']>=entry+tp: pnl=tp; break
        bruto=pnl*n_contratos*20
        neto =bruto-comision*2*n_contratos
        trades.append({'pnl_pts':pnl,'bruto':bruto,'neto':neto,
                       'fecha':str(df.loc[idx,'fecha']),
                       'hora':df.loc[idx,'hour'],
                       'dia':df.loc[idx,'dia_num']})
    return pd.DataFrame(trades)

# ════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 VIXπ-Fusion")
    st.markdown("---")

    st.markdown("### 📂 Datos")
    uploaded = st.file_uploader(
        "Sube tu archivo de datos (Excel 1min)",
        type=['xlsx','xls'],
        help="Archivo con columnas: datetime, open, high, low, close"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Parámetros")
    sl_param = st.slider("Stop Loss (pts)", 8, 20, 12)
    tp_param = st.slider("Take Profit (pts)", 10, 60, 20)
    n_contr  = st.selectbox("Contratos", [1, 2, 3], index=1)
    comision = st.number_input("Comisión $/contrato/lado", 2.0, 10.0, 5.0, 0.5)

    filtro_tipo = st.selectbox(
        "Filtro de señal",
        ['base','thales','thales_atr','pro_thales_atr'],
        index=3,
        format_func=lambda x: {
            'base':          '① Base (horas buenas)',
            'thales':        '② + Thales > 0.5',
            'thales_atr':    '③ + Thales + ATR≤14',
            'pro_thales_atr':'④ + PRO↓ + Thales + ATR≤14 ⭐'
        }[x]
    )

    st.markdown("---")
    st.markdown("### 📊 Tabla día×hora")
    min_trades = st.slider("Mínimo trades por celda", 3, 10, 3)
    mostrar_todo = st.checkbox("Mostrar horas nocturnas", value=True)

    st.markdown("---")
    st.caption("VIXπ-Fusion · Backtest app\nDatos: NQ/MNQ 1min")

# ════════════════════════════════════════════════════
# CARGA DE DATOS
# ════════════════════════════════════════════════════
if uploaded is None:
    st.markdown("# 📈 VIXπ-Fusion · Dashboard")
    st.markdown("### Sube tu archivo de datos para comenzar el análisis")
    col1,col2,col3=st.columns(3)
    with col1:
        st.info("**1.** Sube el Excel de cotizaciones en 1 minuto desde el sidebar izquierdo")
    with col2:
        st.info("**2.** El sistema calcula automáticamente todos los indicadores y señales")
    with col3:
        st.info("**3.** Explora las pestañas: Resumen, Tabla día×hora, Backtest y Señales")
    st.stop()

# Cargar y procesar
with st.spinner("Calculando indicadores..."):
    df_raw = pd.read_excel(uploaded)
    df = calcular_indicador(df_raw)

fecha_ini = df['dt'].min().strftime('%d %b %Y')
fecha_fin = df['dt'].max().strftime('%d %b %Y')
n_velas   = len(df)
n_cruces  = df['cruce'].sum()

# ════════════════════════════════════════════════════
# TABS PRINCIPALES
# ════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Resumen",
    "🗓️ Tabla Día × Hora",
    "📈 Backtest",
    "🔔 Señales recientes"
])

# ════════════════════════════════════════════════════
# TAB 1 — RESUMEN
# ════════════════════════════════════════════════════
with tab1:
    st.markdown(f"### VIXπ-Fusion · {fecha_ini} → {fecha_fin}")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Velas totales",    f"{n_velas:,}")
    c2.metric("Cruces alcistas",  f"{n_cruces}")
    c3.metric("Periodo",          f"{(df['dt'].max()-df['dt'].min()).days} días")
    c4.metric("Último cierre",    f"{df['close'].iloc[-1]:,.2f}")
    c5.metric("ATR actual",       f"{df['atr14'].iloc[-1]:.1f} pts")

    st.markdown("---")

    # Gráfico equity con filtro actual
    with st.spinner("Calculando backtest..."):
        trades_df = backtest_filtrado(df, filtro_tipo, sl_param, tp_param, n_contr, comision)

    if len(trades_df) > 0:
        arr = trades_df['neto'].values
        wins = (arr>0).sum()
        pf = arr[arr>0].sum()/abs(arr[arr<=0].sum()) if (arr<=0).sum()>0 else 99
        eq = np.cumsum(arr)
        dd = (np.maximum.accumulate(np.append([0],eq))-np.append([0],eq)).max()

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Trades",       f"{len(arr)}")
        c2.metric("Win Rate",     f"{wins/len(arr)*100:.1f}%")
        c3.metric("P&L neto",     f"${arr.sum():,.0f}",
                  delta=f"+${arr.sum():,.0f}" if arr.sum()>0 else f"${arr.sum():,.0f}")
        c4.metric("Profit Factor",f"{pf:.2f}")
        c5.metric("Max Drawdown", f"−${dd:,.0f}")

        # Equity curve
        eq_df = pd.DataFrame({'trade': range(1,len(eq)+1), 'equity': eq})
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq_df['trade'], y=eq_df['equity'],
            mode='lines', name='Equity',
            line=dict(color='#2ecc9a', width=2),
            fill='tozeroy',
            fillcolor='rgba(46,204,154,0.08)'
        ))
        fig.update_layout(
            title=f"Curva de equity — {filtro_tipo}",
            paper_bgcolor='#13161e', plot_bgcolor='#13161e',
            font=dict(color='#8b90a0'),
            xaxis=dict(title='Número de trade', gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title='P&L acumulado ($)', gridcolor='rgba(255,255,255,0.05)'),
            height=320, margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Por semana
        trades_df['semana'] = pd.to_datetime(trades_df['fecha']).dt.to_period('W').astype(str)
        sem = trades_df.groupby('semana')['neto'].sum().reset_index()
        sem.columns = ['Semana','P&L neto $']
        colors = ['#2ecc9a' if v>=0 else '#c04444' for v in sem['P&L neto $']]
        fig2 = go.Figure(go.Bar(
            x=sem['Semana'], y=sem['P&L neto $'],
            marker_color=colors, name='P&L por semana'
        ))
        fig2.update_layout(
            title="P&L neto por semana",
            paper_bgcolor='#13161e', plot_bgcolor='#13161e',
            font=dict(color='#8b90a0'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickangle=-45),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            height=280, margin=dict(l=10,r=10,t=40,b=60)
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No hay trades con los filtros seleccionados.")

# ════════════════════════════════════════════════════
# TAB 2 — TABLA DÍA × HORA
# ════════════════════════════════════════════════════
with tab2:
    st.markdown("### Tabla Día × Hora — SL/TP óptimo por combinación")

    with st.spinner("Calculando grid día × hora..."):
        res_df = calcular_grid_dia_hora(df, min_trades=min_trades)

    if len(res_df) == 0:
        st.warning("No hay suficientes datos. Reduce el mínimo de trades por celda.")
        st.stop()

    # Filtros
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        cats_sel = st.multiselect(
            "Categorías",
            ['EXCELENTE','BUENO','ACEPTABLE','MARGINAL','EVITAR'],
            default=['EXCELENTE','BUENO'],
        )
    with col_f2:
        if mostrar_todo:
            hora_rango = st.slider("Rango de horas UTC", 0, 23, (0, 23))
        else:
            hora_rango = st.slider("Rango de horas UTC", 0, 23, (10, 19))

    res_filt = res_df[
        res_df['categoria'].isin(cats_sel) &
        (res_df['hora'] >= hora_rango[0]) &
        (res_df['hora'] <= hora_rango[1])
    ]

    # Stats rápidas
    exc = len(res_filt[res_filt['categoria']=='EXCELENTE'])
    exc_n = len(res_filt[(res_filt['categoria']=='EXCELENTE')&(res_filt['night'])])
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Combinaciones mostradas", len(res_filt))
    c2.metric("Excelentes totales", exc)
    c3.metric("Excelentes nocturnas 🌙", exc_n)
    c4.metric("Excelentes diurnas ☀️", exc-exc_n)

    st.markdown("---")

    # Tabla pivote visual
    DIAS_ORDER = ['Lunes','Martes','Miércoles','Jueves','Viernes']
    cat_emoji  = {'EXCELENTE':'⭐','BUENO':'✅','ACEPTABLE':'🟡','MARGINAL':'⚠️','EVITAR':'❌'}

    # Construir DataFrame pivote con texto enriquecido
    pivot_data = {}
    for dia in DIAS_ORDER:
        col_data = {}
        for hora in range(hora_rango[0], hora_rango[1]+1):
            row = res_filt[(res_filt['dia_nombre']==dia)&(res_filt['hora']==hora)]
            if len(row)==0:
                col_data[hora] = ''
            else:
                r = row.iloc[0]
                night = '🌙' if r['night'] else ''
                pf_str = '>50' if r['best_pf']>50 else f"{r['best_pf']:.2f}"
                col_data[hora] = (f"{cat_emoji.get(r['categoria'],'')} {r['categoria']}{night}\n"
                                  f"SL {r['best_sl']}/TP {r['best_tp']}\n"
                                  f"WR {r['best_wr']:.0f}% · PF {pf_str}\n"
                                  f"{r['n']} trades")
        pivot_data[dia] = col_data

    pivot_df = pd.DataFrame(pivot_data,
        index=range(hora_rango[0], hora_rango[1]+1))
    pivot_df.index.name = 'Hora UTC'

    # Mostrar como tabla estilizada
    def style_cell(val):
        if 'EXCELENTE' in str(val):
            if '🌙' in str(val):
                return 'background-color: rgba(120,100,220,0.2); color: #c4b5fd'
            return 'background-color: rgba(26,122,94,0.25); color: #2ecc9a'
        if 'BUENO' in str(val):
            return 'background-color: rgba(26,122,94,0.10); color: #52c89a'
        if 'ACEPTABLE' in str(val):
            return 'background-color: rgba(220,140,40,0.12); color: #dca028'
        if 'EVITAR' in str(val):
            return 'background-color: rgba(180,50,50,0.10); color: #c04444; opacity: 0.6'
        return 'color: #555a6a'

    styled = pivot_df.style.applymap(style_cell)
    styled = styled.set_properties(**{
        'white-space': 'pre-wrap',
        'font-size': '11px',
        'padding': '8px',
        'border': '1px solid rgba(255,255,255,0.07)',
        'text-align': 'center',
        'vertical-align': 'top',
        'min-width': '130px',
    })
    st.dataframe(styled, height=600, use_container_width=True)

    # Descarga
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        res_filt.to_excel(writer, index=False, sheet_name='Resultados')
    st.download_button(
        "⬇️ Descargar tabla completa (Excel)",
        data=buf.getvalue(),
        file_name=f"vixpi_tabla_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.ms-excel"
    )

# ════════════════════════════════════════════════════
# TAB 3 — BACKTEST DETALLADO
# ════════════════════════════════════════════════════
with tab3:
    st.markdown("### Backtest detallado")

    with st.spinner("Calculando..."):
        trades_df2 = backtest_filtrado(df, filtro_tipo, sl_param, tp_param, n_contr, comision)

    if len(trades_df2) == 0:
        st.warning("Sin trades con estos parámetros.")
        st.stop()

    arr2 = trades_df2['neto'].values
    wins2 = (arr2>0).sum()
    pf2 = arr2[arr2>0].sum()/abs(arr2[arr2<=0].sum()) if (arr2<=0).sum()>0 else 99
    eq2 = np.cumsum(arr2)
    dd2 = (np.maximum.accumulate(np.append([0],eq2))-np.append([0],eq2)).max()
    racha=mr=0
    for v in arr2:
        if v<=0: racha+=1; mr=max(mr,racha)
        else: racha=0

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Trades",       len(arr2))
    c2.metric("Win Rate",     f"{wins2/len(arr2)*100:.1f}%")
    c3.metric("P&L bruto",    f"${(arr2.sum()+comision*2*n_contr*len(arr2)):,.0f}")
    c4.metric("Comisiones",   f"−${comision*2*n_contr*len(arr2):,.0f}")
    c5.metric("P&L neto",     f"${arr2.sum():,.0f}")
    c6.metric("Profit Factor",f"{pf2:.2f}")

    c1b,c2b,c3b = st.columns(3)
    c1b.metric("Max Drawdown",     f"−${dd2:,.0f}")
    c2b.metric("Racha máx pérdidas",f"{mr} trades")
    c3b.metric("Media/trade",      f"${arr2.mean():,.0f}")

    # Por hora
    st.markdown("#### Por hora")
    por_hora = trades_df2.groupby('hora').agg(
        trades=('neto','count'),
        wr=('neto', lambda x: (x>0).mean()*100),
        pnl_neto=('neto','sum')
    ).reset_index()
    por_hora.columns=['Hora UTC','Trades','WR %','P&L neto $']
    por_hora['P&L neto $'] = por_hora['P&L neto $'].round(0)
    por_hora['WR %'] = por_hora['WR %'].round(1)

    colors_hora = ['#2ecc9a' if v>=0 else '#c04444' for v in por_hora['P&L neto $']]
    fig3 = go.Figure(go.Bar(
        x=por_hora['Hora UTC'].astype(str)+'h',
        y=por_hora['P&L neto $'],
        marker_color=colors_hora,
        text=por_hora['WR %'].apply(lambda x: f"WR {x:.0f}%"),
        textposition='outside'
    ))
    fig3.update_layout(
        title="P&L neto por hora",
        paper_bgcolor='#13161e', plot_bgcolor='#13161e',
        font=dict(color='#8b90a0'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        height=320, margin=dict(l=10,r=10,t=40,b=10)
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Tabla de trades
    st.markdown("#### Operaciones individuales")
    display_df = trades_df2[['fecha','hora','pnl_pts','bruto','neto']].copy()
    display_df.columns = ['Fecha','Hora','PnL pts','Bruto $','Neto $']
    display_df['Bruto $'] = display_df['Bruto $'].round(0)
    display_df['Neto $']  = display_df['Neto $'].round(0)

    def color_neto(val):
        if val > 0: return 'color: #2ecc9a'
        if val < 0: return 'color: #c04444'
        return ''

    st.dataframe(
        display_df.style.applymap(color_neto, subset=['Neto $']),
        height=400, use_container_width=True
    )

    # Descarga trades
    buf2 = io.BytesIO()
    with pd.ExcelWriter(buf2, engine='openpyxl') as writer:
        trades_df2.to_excel(writer, index=False, sheet_name='Trades')
    st.download_button(
        "⬇️ Descargar trades (Excel)",
        data=buf2.getvalue(),
        file_name=f"vixpi_trades_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.ms-excel"
    )

# ════════════════════════════════════════════════════
# TAB 4 — SEÑALES RECIENTES
# ════════════════════════════════════════════════════
with tab4:
    st.markdown("### Señales recientes del indicador")

    # Últimos cruces
    cruces_df = df[df['cruce']].tail(20)[
        ['dt','hour','dia_nombre','close','vix_fix','thales_corr','atr14','pend_pro_3']
    ].copy()
    cruces_df.columns=['Datetime','Hora','Día','Precio','VIX Fix','Thales r','ATR14','Pend PRO 3v']
    cruces_df['VIX Fix']   = cruces_df['VIX Fix'].round(1)
    cruces_df['Thales r']  = cruces_df['Thales r'].round(3)
    cruces_df['ATR14']     = cruces_df['ATR14'].round(2)
    cruces_df['Pend PRO 3v'] = cruces_df['Pend PRO 3v'].round(5)

    # Clasificar señal
    def clasif_senal(row):
        t  = np.isfinite(row['Thales r']) and row['Thales r'] > 0.5
        a  = row['ATR14'] <= 14
        p  = row['Pend PRO 3v'] < 0
        h  = 10 <= row['Hora'] <= 19
        if p and t and a and h: return '⭐ Excelente'
        if t and a and h:       return '✅ Buena'
        if t and h:             return '🟡 Aceptable'
        return '⚪ Débil'

    cruces_df['Calidad'] = cruces_df.apply(clasif_senal, axis=1)

    def color_calidad(val):
        if '⭐' in str(val): return 'color: #2ecc9a; font-weight: 700'
        if '✅' in str(val): return 'color: #52c89a'
        if '🟡' in str(val): return 'color: #dca028'
        return 'color: #555a6a'

    st.dataframe(
        cruces_df.style.applymap(color_calidad, subset=['Calidad']),
        height=500, use_container_width=True
    )

    # Estado actual del indicador
    st.markdown("---")
    st.markdown("#### Estado actual del indicador")
    last = df.iloc[-1]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("EMA PRO",   f"{last['ema_pro']:.4f}")
    c2.metric("EMA 1/7",   f"{last['ema_17']:.4f}")
    c3.metric("Thales r",  f"{last['thales_corr']:.3f}",
              delta="✅ coherente" if last['thales_corr']>0.5 else "⚠️ bajo")
    c4.metric("ATR 14",    f"{last['atr14']:.1f} pts",
              delta="✅ OK" if last['atr14']<=14 else "⚠️ alto")

    ema17_sobre = last['ema_17'] > last['ema_pro']
    st.info(f"**Posición EMAs:** {'🟢 EMA 1/7 POR ENCIMA de EMA PRO — ventana alcista abierta' if ema17_sobre else '⚪ EMA 1/7 por debajo de EMA PRO — sin señal activa'}")
