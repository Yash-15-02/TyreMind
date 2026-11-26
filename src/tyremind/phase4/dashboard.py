"""Display the saved fleet snapshot and investigation inbox in Streamlit."""
from collections import Counter
import sqlite3

import plotly.graph_objects as go
import streamlit as st

from tyremind.phase4.technician import record_service, record_completion
from tyremind.phase4.improvements import save_improvements
from tyremind.phase4.evaluate import evaluate
from tyremind.phase4.review import record_review
from tyremind.phase4.read import read_results, read_tyre_history


COLORS = {'Normal': '#268477', 'Medium': '#BF891E', 'High': '#CE623B',
          'Critical': '#A33045', 'Unknown': '#778795'}


def label(key):
    return key.replace('_', ' ').capitalize().replace('kpa', 'kPa').replace('sha256', 'SHA-256')


def display_value(value):
    if value is None:
        return 'Unavailable'
    if isinstance(value, bool):
        return 'Yes' if value else 'No'
    if isinstance(value, float):
        return f'{value:,.3f}'.rstrip('0').rstrip('.')
    return str(value)


def detail_rows(values, prefix=''):
    rows = []
    for key, value in values.items():
        name = prefix + label(key)
        if isinstance(value, dict):
            rows.extend(detail_rows(value, name + ' · '))
        elif isinstance(value, list):
            for entry in value:
                rows.append({'Detail': name, 'Value': display_value(entry)})
        else:
            rows.append({'Detail': name, 'Value': display_value(value)})
    return rows


def show_measurements(evidence):
    cols = st.columns(3)
    for col, (name, key) in zip(cols, [('Pressure (kPa)', 'pressure_kpa'),
                                       ('Temperature (°C)', 'temperature_c'),
                                       ('Measured tread (mm)', 'latest_measured_tread_mm')]):
        col.metric(name, display_value(evidence.get(key)))
    st.caption('Reference pressure (kPa): ' + display_value(evidence.get('reference_pressure_kpa'))
               + ' · Days since tread inspection: ' + display_value(evidence.get('days_since_inspection')))
    prediction = evidence.get('prediction') or {}
    st.subheader('Model results')
    if prediction:
        cols = st.columns(3)
        cols[0].metric('Forecast tread at +5,000 km (mm)', display_value(prediction.get('predicted_tread_depth_5000km_mm')))
        cols[1].metric('Anomaly score', display_value(prediction.get('anomaly_score')))
        cols[2].metric('Anomaly threshold', display_value(prediction.get('anomaly_threshold')))
        flag = {1: 'Flagged', 0: 'Within threshold'}.get(prediction.get('anomaly_flag'), 'Unavailable')
        st.caption(f"Wear model: {display_value(prediction.get('wear_model'))} · Anomaly result: {flag}")
    else:
        st.caption('No saved prediction for this assessment.')
    if evidence.get('feature_values'):
        st.dataframe(detail_rows(evidence['feature_values']), hide_index=True, use_container_width=True)


def show_outcome(message, allow_review=False):
    outcome = message['outcome']
    failed = outcome['step'] == 12
    title = 'Revision required' if failed else 'Awaiting human review'
    review = message.get('review')
    if review:
        title = review['decision'].replace('_', ' ').title()
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
        elif outcome.get('review_only'):
            st.write(outcome['report']['reason'])
            for check in outcome['report']['checks']:
                st.write(f'• {check}')
            st.table(detail_rows(outcome['alert']))
        else:
            for label, text in [('Health assessment', outcome['report']['assessment']),
                                ('Root-cause hypotheses', outcome['report']['analysis']),
                                (('Approved work order' if review and review['decision'] == 'APPROVED' else 'Draft work order'), outcome['work_order']['instructions'])]:
                st.subheader(label)
                st.write(text)
            st.table(detail_rows(outcome['alert']))
        for note in validation['evidence_notes']:
            st.info(note)
        for ref in validation['references']:
            st.markdown(f"[{ref['reference']}]({ref['url']})")

        if review:
            st.write(f"Reviewed by {review['reviewer']} · {review['reviewed_at']} UTC")
            if review['note']:
                st.text(review['note'])
        elif allow_review and not failed and not outcome.get('review_only'):
            with st.form(f"review-{outcome['investigation_id']}"):
                st.subheader('Fleet manager review')
                name = st.text_input('Reviewer name')
                decision = st.selectbox('Approve draft?', ['APPROVED', 'REJECTED'], index=None,
                                        format_func={'APPROVED': 'Yes — approve', 'REJECTED': 'No — reject'}.get)
                note = st.text_area('Review note (reason required for rejection)')
                submit = st.form_submit_button('Save decision')
            if submit:
                try:
                    record_review(outcome['investigation_id'], message['payload_sha256'], name, decision, note)
                except (ValueError, sqlite3.Error) as exc:
                    st.error(str(exc))
                else:
                    st.rerun()


def show_tyre(current, inbox):
    tyre = current['tyre_id']
    st.subheader(f"{tyre} · {current['risk_level'] or 'Unknown'} risk")
    st.caption(f"Latest reading {current['timestamp']} · Evidence cutoff {current['feature_as_of']}")
    history = read_tyre_history(tyre, current['feature_as_of'])
    metrics = {'Pressure (kPa)': ('telemetry', 'pressure_kpa'),
               'Temperature (°C)': ('telemetry', 'temperature_c'),
               'Load ratio': ('telemetry', 'load_ratio'),
               'Measured tread depth (mm)': ('inspections', 'tread_depth_mm')}
    label = st.radio('Measurement trend', list(metrics), horizontal=True)
    table, field = metrics[label]
    readings = history[table]
    if readings:
        figure = go.Figure(go.Scatter(x=[r['timestamp'] for r in readings],
                                      y=[r[field] for r in readings], mode='lines+markers'))
        figure.update_layout(yaxis_title=label, xaxis_title='Observation date', height=300)
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.info('No recorded measurements for this trend.')
    risks = history['risk']
    levels = [r['risk_level'] or 'Unknown' for r in risks]
    timeline = go.Figure(go.Scatter(x=[r['timestamp'] for r in risks], y=levels, mode='markers',
                                    marker=dict(size=10, color=[COLORS.get(v, COLORS['Unknown']) for v in levels]),
                                    text=[r['data_status'] for r in risks]))
    timeline.update_layout(title='Recorded risk assessments', height=270, xaxis_title='Assessment cutoff',
                           yaxis=dict(type='category', categoryorder='array',
                                      categoryarray=['Unknown', 'Normal', 'Medium', 'High', 'Critical']))
    st.plotly_chart(timeline, use_container_width=True)
    st.subheader('AI findings and source references')
    matches = [m for m in reversed(inbox) if m['outcome']['tyre_id'] == tyre]
    for message in matches:
        show_outcome(message)
    if not matches:
        st.caption('No published investigation for this tyre.')


def technician_view(inbox):
    orders = {m['outcome']['investigation_id']: m for m in inbox
              if m.get('approved_order') and (m.get('review') or {}).get('decision') == 'APPROVED'}
    if not orders:
        st.info('No approved work orders are available for inspection.')
        return
    identity = st.selectbox('Approved work order', list(orders),
                            format_func=lambda key: f"{orders[key]['approved_order']['tyre_id']} · {key}")
    message = orders[identity]
    order = message['approved_order']
    st.caption(f"Approved by {order['approved_by']} · {order['approved_at']} UTC")
    st.subheader('Authorized instructions')
    st.write(order['instructions'])
    service = message['service']
    if service:
        st.caption(f"Recorded by {service['technician']} · {service['recorded_at']} UTC")
        st.write('Inspection findings', service['findings'])
        st.write('Maintenance performed', service['maintenance'] or 'None recorded')
        latest = message.get('completion')
        st.subheader('Inspection outcome')
        if latest:
            st.write(latest['status'].replace('_', ' ').title())
            st.text(latest['actual_findings'])
            st.caption(f"{latest['recorded_by']} · {latest['recorded_at']} UTC")
        if not latest or latest['status'] != 'COMPLETED':
            with st.form(f"outcome-{identity}-{latest['id'] if latest else 0}"):
                recorder = st.text_input('Recorded by', value=service['technician'])
                findings = st.text_area('Actual findings', value=latest['actual_findings'] if latest else service['findings'])
                status = st.selectbox('Work order status', ['COMPLETED', 'FOLLOW_UP_REQUIRED'], index=None,
                                      format_func=lambda value: value.replace('_', ' ').title())
                save = st.form_submit_button('Save inspection outcome')
            if save:
                try:
                    record_completion(identity, message['payload_sha256'], findings, status, recorder,
                                      latest['id'] if latest else None)
                except (ValueError, sqlite3.Error) as exc:
                    st.error(str(exc))
                else:
                    st.rerun()
        return
    with st.form(f'technician-{identity}'):
        name = st.text_input('Technician name')
        findings = st.text_area('Inspection findings')
        outcome = st.radio('Maintenance outcome', ['Inspection only', 'Authorized maintenance performed'], index=None)
        details = st.text_area('Authorized maintenance actually performed')
        submit = st.form_submit_button('Save technician record')
    if submit:
        try:
            performed = None if outcome is None else outcome == 'Authorized maintenance performed'
            record_service(identity, message['payload_sha256'], name, findings, performed, details)
        except (ValueError, sqlite3.Error) as exc:
            st.error(str(exc))
        else:
            st.rerun()


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
    overview, cases, technician, evaluation = st.tabs(['Fleet monitoring', f'Investigation inbox ({len(inbox)})', 'Technician work orders', 'Evaluation'])
    with evaluation:
        if st.button('Evaluate saved results'):
            try:
                report = evaluate()
            except (ValueError, sqlite3.Error) as exc:
                st.error(str(exc))
            else:
                for key, title in [('alerts', 'Alert detection'), ('tread_forecasts', 'Tread forecasts'),
                                   ('recommendations', 'Recommendation outcomes')]:
                    st.subheader(title)
                    st.dataframe(detail_rows(report[key]), hide_index=True, use_container_width=True)
                st.caption('Null means insufficient observations. Manager decisions and completion track workflow outcomes.')
        st.subheader('Future improvements')
        if st.button('Save improvement report'):
            try:
                saved = save_improvements()
                st.success(f"Saved report · {len(saved['improvements'])} proposals")
                for item in saved['improvements']:
                    with st.container(border=True):
                        st.caption(f"{item['priority']} priority")
                        st.markdown(f"**{item['topic']}**")
                        st.write(item['action'])
                        evidence = item['evidence']
                        if isinstance(evidence, dict):
                            st.table(detail_rows(evidence))
                        else:
                            st.caption(evidence)
            except (ValueError, sqlite3.Error) as exc:
                st.error(str(exc))
    with technician:
        technician_view(inbox)
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
            show_tyre(current, inbox)
            for rule in current['rules']:
                with st.expander(f"{rule['level']} · {label(rule['rule_id'])}"):
                    st.table(detail_rows(rule['values']))
            if current['data_issues']:
                st.warning(' · '.join(current['data_issues']))
            with st.expander('Measurements and model results'):
                show_measurements(current['evidence'])
        else:
            st.info('No monitoring assessments are available.')
    with cases:
        selection = st.selectbox('Investigation tyre', ['All tyres'] + sorted({m['outcome']['tyre_id'] for m in inbox}))
        for message in reversed(inbox):
            if selection == 'All tyres' or message['outcome']['tyre_id'] == selection:
                show_outcome(message, allow_review=True)
        if not inbox:
            st.info('No investigation outcomes have been published yet.')


if __name__ == '__main__':
    main()
