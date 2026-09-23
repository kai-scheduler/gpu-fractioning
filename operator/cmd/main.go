// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"crypto/tls"
	"os"

	_ "k8s.io/client-go/plugin/pkg/client/auth"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	v1alpha1 "github.com/kai-scheduler/gpu-fractioning/api/v1alpha1"
	"github.com/kai-scheduler/gpu-fractioning/operator/internal/common/daemonmgr"
	"github.com/kai-scheduler/gpu-fractioning/operator/internal/config"
	"github.com/kai-scheduler/gpu-fractioning/operator/internal/controller"
	"github.com/kai-scheduler/gpu-fractioning/pkg/env"
	// +kubebuilder:scaffold:imports
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(v1alpha1.AddToScheme(scheme))
	// +kubebuilder:scaffold:scheme
}

func main() {
	cfg := config.ParseFlags()

	ctrl.SetLogger(zap.New(zap.UseDevMode(cfg.Development)))

	// ── TLS configuration ────────────────────────────────────────────────
	// HTTP/2 is disabled by default to mitigate the Rapid Reset CVE.
	var tlsOpts []func(*tls.Config)
	if !cfg.EnableHTTP2 {
		tlsOpts = append(tlsOpts, func(c *tls.Config) {
			c.NextProtos = []string{"http/1.1"}
		})
	}

	// ── Metrics server ───────────────────────────────────────────────────
	metricsServerOptions := metricsserver.Options{
		BindAddress:   cfg.MetricsAddr,
		SecureServing: cfg.SecureMetrics,
		TLSOpts:       tlsOpts,
	}
	if cfg.SecureMetrics {
		metricsServerOptions.FilterProvider = filters.WithAuthenticationAndAuthorization
	}
	if len(cfg.MetricsCertPath) > 0 {
		metricsServerOptions.CertDir = cfg.MetricsCertPath
		metricsServerOptions.CertName = cfg.MetricsCertName
		metricsServerOptions.KeyName = cfg.MetricsCertKey
	}

	// ── Operator namespace ───────────────────────────────────────────────
	podNamespace := os.Getenv("POD_NAMESPACE")
	if podNamespace == "" {
		podNamespace = "default"
	}
	setupLog.Info("operator namespace", "namespace", podNamespace)

	// ── Controller manager ───────────────────────────────────────────────
	// We do not configure a Pod (or Node) cache. The controller reads Pods and
	// Nodes only in the rare unhealthy/recovery path (node-condition patching)
	// and does so via the manager's uncached API reader, so it never maintains
	// cluster-scale Pod/Node informers. Reconciles are driven by GpuFractioningConfig
	// and DaemonSet/dependency metadata events, not pod events. Only DaemonSets,
	// the CR, and metadata for GPU Operator dependency resources — small, bounded
	// object sets — are served from the default cache.
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:                 scheme,
		Metrics:                metricsServerOptions,
		HealthProbeBindAddress: cfg.ProbeAddr,
		LeaderElection:         cfg.EnableLeaderElect,
		LeaderElectionID:       "gpu-fractioning.kai.scheduler",
	})
	if err != nil {
		setupLog.Error(err, "Failed to start manager")
		os.Exit(1)
	}

	// ── Component images (from Helm-injected env vars) ──────────────────
	fractiondImage := controller.ReadImageFromEnv("FRACTIOND_IMAGE")
	metricsdImage := controller.ReadImageFromEnv("METRICSD_IMAGE")
	mpsdImage := controller.ReadImageFromEnv("MPSD_IMAGE")

	for name, img := range map[string]daemonmgr.ImageSpec{
		"fractiond": fractiondImage,
		"metricsd":  metricsdImage,
		"mpsd":      mpsdImage,
	} {
		if img.FullImage() == "" {
			setupLog.Error(nil, "component image is not configured; set the corresponding Helm value", "component", name)
			os.Exit(1)
		}
	}
	setupLog.Info("fractiond default image", "image", fractiondImage.FullImage())
	setupLog.Info("metricsd default image", "image", metricsdImage.FullImage())
	setupLog.Info("mpsd default image", "image", mpsdImage.FullImage())

	// ── mpsd MPS config (Helm-injected default; forwarded to the mpsd pod) ──
	mpsdAuditLog := env.Bool("MPSD_AUDIT_LOG", true)
	setupLog.Info("mpsd MPS memacct audit log", "enabled", mpsdAuditLog)

	// ── sm-sharing chicken bit (Helm-injected; forwarded to mpsd and fractiond) ──
	// A kill switch for the whole sm-sharing compute mode: disabling it turns
	// off mpsd's shared MPS server and makes fractiond reject the
	// gpu-compute.mode: sm-sharing annotation, without a code rollback.
	supportSMSharing := env.Bool("SUPPORT_SM_SHARING", true)
	setupLog.Info("sm-sharing compute mode support", "enabled", supportSMSharing)

	// ── GPU stack version-check bypass (Helm-injected) ───────────────────
	// A revert lever for the dependency version gates: when the operator reads
	// an installation's GPU stack versions wrongly and refuses to roll out, this
	// unblocks it without a code rollback. It disables verification only — it
	// does not make an unsupported GPU stack work.
	skipGPUStackVersionChecks := env.Bool("SKIP_GPU_STACK_VERSION_CHECKS", false)
	if skipGPUStackVersionChecks {
		// logr has no warning level, so a nil-error Error call is how this
		// codebase raises one (see the component image check above).
		setupLog.Error(nil, "GPU stack version checks are bypassed; dependency versions will not be verified. "+
			"An unsupported GPU stack now rolls out silently instead of being reported on the Ready condition")
	}

	// ── FIPS mode (Helm-injected; forwarded to every daemon container) ──
	// Carried as the chart's own "off"/"on"/"only" vocabulary rather than a bool
	// so the operator's pod spec states the installation's compliance posture
	// plainly. Only "only" changes anything the operator does: what makes a
	// binary FIPS-compliant is the validated module linked into the image, and
	// the chart selects those images from this same value.
	fipsMode := env.String("FIPS_MODE", "off")
	fipsOnly := fipsMode == "only"
	setupLog.Info("FIPS mode", "mode", fipsMode, "enforcement", fipsOnly)

	// ── Daemon pod API identity (Helm-injected; used by mpsd startup labeling) ──
	daemonServiceAccountName := env.String("DAEMON_SERVICE_ACCOUNT_NAME", "")
	setupLog.Info("daemon service account", "serviceAccountName", daemonServiceAccountName)

	// ── Register controllers ─────────────────────────────────────────────
	if err := controller.NewGpuFractioningConfigReconciler(
		mgr.GetClient(),
		mgr.GetAPIReader(),
		mgr.GetScheme(),
		//nolint:staticcheck // record.EventRecorder is still supported; migrating the reconciler to the new events API is tracked separately.
		mgr.GetEventRecorderFor("gpufractioningconfig-controller"),
		podNamespace,
		map[string]daemonmgr.ImageSpec{
			"fractiond": fractiondImage,
			"metricsd":  metricsdImage,
			"mpsd":      mpsdImage,
		},
		daemonServiceAccountName,
		mpsdAuditLog,
		supportSMSharing,
		fipsOnly,
		skipGPUStackVersionChecks,
	).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "Failed to create controller", "controller", "gpufractioningconfig")
		os.Exit(1)
	}
	// +kubebuilder:scaffold:builder

	// ── Health probes ────────────────────────────────────────────────────
	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "Failed to set up health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "Failed to set up ready check")
		os.Exit(1)
	}

	// ── Start ────────────────────────────────────────────────────────────
	setupLog.Info("Starting manager")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "Failed to run manager")
		os.Exit(1)
	}
}
