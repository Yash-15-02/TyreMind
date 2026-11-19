"""Display the saved fleet snapshot and investigation inbox in Streamlit."""
from collections import Counter
import sqlite3

import plotly.graph_objects as go
import streamlit as st

from tyremind.phase4.read import read_results


COLORS = {'Normal': '#268477', 'Medium': '#BF891E', 'High': '#CE623B',
          'Critical': '#A33045', 'Unknown': '#778795'}


def show_outcome(message):
    outcome = message['outcome']
    failed = outcome['step'] == 12
    title = 'Revision required' if failed else 'Awaiting human review'
    with st.expander(f"{outcome['tyre_id']} · {title} · {message['published_at']}"):
        validation = outcome['validation']
        st.caption(f"{validation['priority']} priority · Evidence cutoff {validation['cutoff']}")
        if failed:
            st.error(outcome['error'])
            for finding in validation['findings']:
                st.write(f"**{finding['section'].title()} · {finding['code']}**")
                st.write(finding['detail'])
                if finding.get('excerpt'):
                    st.text(finding['excerpt'])
        else:
            for label, text in [('Health assessment', outcome['report']['assessment']),
                                ('Root-cause hypotheses', outcome['report']['analysis']),
                                ('Draft work order', outcome['work_order']['instructions'])]:
                st.subheader(label)
                st.write(text)
            st.write('Alert', outcome['alert'])
        for note in validation['evidence_notes']:
            st.info(note)
        for ref in validation['references']:
            st.markdown(f"[{ref['reference']}]({ref['url']})")


def main():
    st.set_page_config(page_title='TyreMind | Fleet dashboard', layout='wide')
    st.title('Fleet dashboard')
    st.sidebar.header('TyreMind')
    st.sidebar.button('Refresh from SQLite')
    try:
        snapshot = read_results()
    except (sqlite3.Error, ValueError) as exc:
        st.error(f'Cannot load fleet data: {exc}. Run from the project folder with its monitoring database.')
        return
    rows = snapshot['monitoring']
    inbox = snapshot['investigations']
    overview, cases = st.tabs(['Fleet monitoring', f'Investigation inbox ({len(inbox)})'])
    with overview:
        truck = st.selectbox('Truck', ['All trucks'] + sorted({r['truck_id'] for r in rows}))
        visible = [r for r in rows if truck == 'All trucks' or r['truck_id'] == truck]
        counts = Counter(r['risk_level'] or 'Unknown' for r in visible)
        a, b, c = st.columns(3)
        a.metric('Monitored tyres', len(visible))
        b.metric('High risk', counts['High'])
        c.metric('Critical risk', counts['Critical'])
        if visible:
            st.caption('Latest assessment per tyre · Most recent evidence cutoff ' +
                       max(r['feature_as_of'] for r in visible))
            chart = go.Figure(go.Bar(x=list(COLORS), y=[counts[level] for level in COLORS],
                                     marker_color=list(COLORS.values())))
            chart.update_layout(title='Tyres by risk level', yaxis_title='Tyres', height=300,
                                margin=dict(t=50,b=25))
            st.plotly_chart(chart, use_container_width=True)
            st.dataframe([dict(Tyre=r['tyre_id'], Truck=r['truck_id'], Risk=r['risk_level'] or 'Unknown',
                               Evidence=r['data_status'], Reading=r['timestamp']) for r in visible],
                         hide_index=True, use_container_width=True)
            by_id = {r['tyre_id']: r for r in visible}
            current = by_id[st.selectbox('Inspect tyre', list(by_id))]
            st.write('Triggered rules', current['rules'])
            if current['data_issues']:
                st.warning(' · '.join(current['data_issues']))
            with st.expander('Measurements and model results'):
                st.json(current['evidence'])
        else:
            st.info('No monitoring assessments are available.')
    with cases:
        selection = st.selectbox('Investigation tyre', ['All tyres'] + sorted({m['outcome']['tyre_id'] for m in inbox}))
        for message in reversed(inbox):
            if selection == 'All tyres' or message['outcome']['tyre_id'] == selection:
                show_outcome(message)
        if not inbox:
            st.info('No investigation outcomes have been published yet.')


if __name__ == '__main__':
    main()
