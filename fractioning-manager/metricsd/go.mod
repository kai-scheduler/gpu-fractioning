module github.com/kai-scheduler/gpu-fractioning/fractioning-manager/metricsd

go 1.26.4

require (
	github.com/NVIDIA/go-nvml v0.13.3-1
	github.com/kai-scheduler/gpu-fractioning v0.0.0
	github.com/kai-scheduler/gpu-fractioning/pkg v0.0.0
	github.com/prometheus/client_golang v1.24.1
	github.com/prometheus/client_model v0.6.2
)

require (
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822 // indirect
	github.com/pmezard/go-difflib v1.0.1-0.20181226105442-5d4384ee4fb2 // indirect
	github.com/prometheus/common v0.70.1 // indirect
	github.com/prometheus/procfs v0.21.1 // indirect
	golang.org/x/sys v0.47.0 // indirect
	google.golang.org/protobuf v1.36.12-0.20260120151049-f2248ac996af // indirect
)

replace github.com/kai-scheduler/gpu-fractioning => ../../

replace github.com/kai-scheduler/gpu-fractioning/pkg => ../../pkg
