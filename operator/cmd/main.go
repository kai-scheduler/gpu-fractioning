// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/tls"
	"os"
	"time"

	_ "k8s.io/client-go/plugin/pkg/client/auth"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	v1alpha1 "github.com/kai-scheduler/kai-gpu-fractioning/api/v1alpha1"
	"github.com/kai-scheduler/kai-gpu-fractioning/operator/internal/common/daemonmgr"
	"github.com/kai-scheduler/kai-gpu-fractioning/operator/internal/config"
	"github.com/kai-scheduler/kai-gpu-fractioning/operator/internal/controller"
	"github.com/kai-scheduler/kai-gpu-fractioning/pkg/env"
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
	restConfig := ctrl.GetConfigOrDie()
	mgr, err := ctrl.NewManager(restConfig, ctrl.Options{
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
		cfg.MinGPUOperatorVersion,
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
	setupLog.Info("Starting manager", "minGPUOperatorVersion", cfg.MinGPUOperatorVersion)
	runErr := mgr.Start(ctrl.SetupSignalHandler())

	// ── Shutdown ─────────────────────────────────────────────────────────
	// Nothing else expires the per-node Ready condition, so an operator that
	// just stops (scaled to zero, evicted, rolled) leaves every node asserting a
	// readiness no one is maintaining. Hand that assertion back on the way out.
	if cfg.MarkNodesUnknownOnShutdown {
		markNodesUnavailable(restConfig, wasElected(mgr))
	}

	if runErr != nil {
		setupLog.Error(runErr, "Failed to run manager")
		os.Exit(1)
	}
}

// shutdownMarkTimeout bounds the post-Start node marking. The pod's termination
// grace period is what actually cuts us off, so this only has to be short
// enough to leave room for the process to exit cleanly inside it.
const shutdownMarkTimeout = 15 * time.Second

// wasElected reports whether this manager ever won leader election. A manager
// that never led was not the one maintaining the node conditions, so it must
// not invalidate them on its way out — that would flap every node each time a
// standby replica restarts. With leader election disabled the channel is closed
// at startup, so a single-replica install always reports true.
func wasElected(mgr ctrl.Manager) bool {
	select {
	case <-mgr.Elected():
		return true
	default:
		return false
	}
}

// markNodesUnavailable flips the gpu-fractioning Ready condition to Unknown on
// every targeted node. It builds its own direct client: the manager has already
// stopped by this point, so its cached client is no longer being served.
func markNodesUnavailable(restConfig *rest.Config, elected bool) {
	if !elected {
		setupLog.Info("skipping shutdown node marking: this replica was never the leader")
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), shutdownMarkTimeout)
	defer cancel()

	c, err := client.New(restConfig, client.Options{Scheme: scheme})
	if err != nil {
		setupLog.Error(err, "Failed to build client for shutdown node marking")
		return
	}

	if err := controller.MarkNodesUnavailable(ctx, c, c); err != nil {
		setupLog.Error(err, "Failed to mark nodes unavailable on shutdown")
		return
	}
	setupLog.Info("marked gpu-fractioning node conditions Unknown for shutdown")
}
