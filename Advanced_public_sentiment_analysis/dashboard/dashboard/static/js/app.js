// Initialize Vue.js application
const app = new Vue({
    el: '#app',
    data: {
        sentimentData: [],
        isLoading: true,
        darkMode: false,
        socket: null,
        charts: {},
        user: null,
        subscription: null
    },
    
    created() {
        // Initialize WebSocket connection
        this.initializeWebSocket();
        // Load user data and check authentication
        this.loadUserData();
        // Initialize charts
        this.initializeCharts();
    },

    methods: {
        initializeWebSocket() {
            this.socket = io({
                auth: {
                    token: localStorage.getItem('token')
                }
            });

            this.socket.on('sentiment_update', (data) => {
                this.updateCharts(data);
            });
        },

        async loadUserData() {
            try {
                const response = await fetch('/api/user', {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('token')}`
                    }
                });
                this.user = await response.json();
                this.loadSubscriptionData();
            } catch (error) {
                console.error('Failed to load user data:', error);
            }
        },

        initializeCharts() {
            // Sentiment trend chart
            this.charts.sentiment = new Chart(
                document.getElementById('sentimentTrend'),
                {
                    type: 'line',
                    data: {
                        datasets: [{
                            label: 'Sentiment Trend',
                            borderColor: '#4CAF50',
                            data: []
                        }]
                    },
                    options: {
                        responsive: true,
                        animation: {
                            duration: 1000,
                            easing: 'easeInOutQuart'
                        },
                        scales: {
                            x: {
                                type: 'time',
                                time: {
                                    unit: 'hour'
                                }
                            },
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                }
            );
        },

        updateCharts(data) {
            // Update sentiment trend
            this.charts.sentiment.data.datasets[0].data.push({
                x: new Date(data.timestamp),
                y: data.sentiment_score
            });
            this.charts.sentiment.update('quiet');
        },

        async subscribe() {
            try {
                const stripe = Stripe(process.env.STRIPE_PUBLIC_KEY);
                const { error } = await stripe.redirectToCheckout({
                    items: [{ plan: 'premium_monthly', quantity: 1 }],
                    successUrl: `${window.location.origin}/subscription/success`,
                    cancelUrl: `${window.location.origin}/subscription/cancel`,
                });
                
                if (error) throw error;
            } catch (error) {
                console.error('Subscription failed:', error);
            }
        },

        toggleDarkMode() {
            this.darkMode = !this.darkMode;
            document.body.classList.toggle('dark-mode');
        }
    }
});
