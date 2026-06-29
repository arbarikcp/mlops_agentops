{{/*
Expand the name of the chart.
*/}}
{{- define "credit-risk.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "credit-risk.fullname" -}}
{{- printf "%s" (include "credit-risk.name" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "credit-risk.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{ include "credit-risk.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels.
*/}}
{{- define "credit-risk.selectorLabels" -}}
app.kubernetes.io/name: {{ include "credit-risk.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
