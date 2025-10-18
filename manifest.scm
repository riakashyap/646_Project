;;; defines an environment for guix
(use-modules (gnu packages python)
	     (gnu packages python-build)
	     (gnu packages jupyter))

(packages->manifest
 (list
  ;; env
  python
  jupyter

  ;; libs
  python-pip
  ))
