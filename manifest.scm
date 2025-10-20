;;; defines an environment for guix
(use-modules (gnu packages jupyter)
             (gnu packages machine-learning)
             (gnu packages python)
             (gnu packages python-build)
             (gnu packages python-xyz))

(packages->manifest
 (list
  ;; env
  python
  python-ipython
  jupyter

  ;; libs
  python-pip
  llama-cpp
  ))
